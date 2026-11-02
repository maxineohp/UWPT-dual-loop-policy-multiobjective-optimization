# -*- coding: utf-8 -*-
"""
train_dynamic_policy_small_dpc.py

用途
----
使用两个已经训练好的网络，训练 WPT 动态相角策略网络：

1. 稳态最优角网络：
       (Vin, Rload, preference) -> (a*, b*, c*)
   只在每个训练场景开始时计算一次目标相角，滚动过程中保持不变。

2. 动态代理模型：
       [Vout, Ip_rms, Is_rms, a, b, c,
        da, db, dc, Vin_new, Rload_new]
       ->
       [dVout, dIp_rms, dIs_rms]

3. 动态策略网络：
       当前闭环状态 + 新稳态目标角 + 电压误差积分
       ->
       [da, db, dc]

动态策略网络通过冻结的动态代理模型进行多步可微滚动训练，不需要
真实“最优动作”标签。代码包含：
- 基于限幅斜坡控制的模仿预训练；
- 多步 DPC 训练；
- 电压误差积分状态；
- 相角、相角变化率、电流和代理模型适用域约束；
- 训练/验证按轨迹文件划分；
- 自动保存最佳模型和验证滚动结果。

默认目录结构
------------
当前脚本同路径下：

steady_policy_nsga_pref_no_pout_output1/
    steady_policy_nsga_pref.pth
    steady_policy_input_scaler.pkl
    steady_policy_angle_bounds.npy

dynamic_surrogate_output/
    dynamic_surrogate.pth
    x_scaler.pkl
    y_scaler.pkl
    split_info.json            # 可选

traj_outputs/*.csv  （按文件划分训练集和验证集）

重要说明
--------
1. CONFIG["Vref"] 默认 200 V，请按你的实际给定值修改。
2. Ip_limit、Is_limit 最好填写实物允许值；若为 None，代码从训练数据分位数估计。
3. 动态代理模型必须先通过多步滚动验证，否则策略可能利用代理误差。
4. 动态策略网络训练后仍需回到 PLECS 闭环验证。
5. 此版本从零训练小模型，不加载以前的教师/学生动态策略权重。
"""

from __future__ import annotations

import json
import math
import pickle
import random
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. 配置
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # 文件路径
    # --------------------------------------------------------
    "steady_policy_dir": "steady_policy_nsga_pref_no_pout_output1",
    "steady_policy_pth": "steady_policy_nsga_pref.pth",
    "steady_policy_scaler": "steady_policy_input_scaler.pkl",
    "steady_policy_angle_bounds": "steady_policy_angle_bounds.npy",

    "dynamic_model_dir": "dynamic_surrogate_output",
    "dynamic_model_pth": "dynamic_surrogate.pth",
    "dynamic_x_scaler": "x_scaler.pkl",
    "dynamic_y_scaler": "y_scaler.pkl",
    "dynamic_split_info": "split_info.json",

    "data_dirs": [
        "traj_outputs",

    ],

    "output_dir": "dynamic_policy_small_dpc_output",

    # --------------------------------------------------------
    # 稳态最优角网络
    # --------------------------------------------------------
    # 如果稳态网络输入是 [Vin, Rload, pref_eff]，使用该偏好量。
    "pref_value": 1,

    # auto / direct_angles / normalized01_to_bounds / sigmoid_to_bounds
    "steady_output_mode": "auto",

    # --------------------------------------------------------
    # 控制设置
    # --------------------------------------------------------
    "Vref": 200,              # 必须按实际输出电压给定值修改
    "Ts_ctrl": 200e-6,

    # 每个控制周期最大相角变化量，单位：deg
    "du_max": [3.0, 3.0, 3.0],  # 训练/部署统一为 +/-3 deg

    # 相角硬边界
    "angle_low": [90.0, 90.0, 90.0],
    "angle_high": [180.0, 180.0, 150.0],

    # 电压误差积分状态 z(k+1)=clip(z(k)+Ts*e(k))
    "integral_min": -0.50,
    "integral_max": 0.50,

    # 电流上限。建议填写实际硬件允许值。
    # None 时使用训练数据 99.5% 分位数 × current_limit_margin。
    "Ip_limit": None,
    "Is_limit": None,
    "current_limit_quantile": 0.995,
    "current_limit_margin": 1.5,

    # --------------------------------------------------------
    # 策略网络
    # 输入共 20 维，输出 3 维
    # --------------------------------------------------------
    "policy_hidden_dims": [64, 64, 32],  # 小网络，从零训练，20->64->64->32->3
    "policy_dropout": 0.0,

    # --------------------------------------------------------
    # 训练场景
    # --------------------------------------------------------
    # 主要场景：从真实轨迹中提取 Vin/Rload 发生阶跃的时刻。
    # 另加入一定比例工况保持场景，使网络学习到稳态附近停止动作。
    "hold_scenario_ratio": 0.20,

    # 训练工况过滤：仅保留 Vin=180~250 V、Rload=80~130 Ω
    "condition_filter_enabled": True,
    "filter_vin_min": 180.0,
    "filter_vin_max": 250.0,
    "filter_rload_min": 80.0,
    "filter_rload_max": 130.0,

    # 若 split_info.json 不可用，则按文件随机划分。
    "fallback_train_ratio": 0.80,
    "fallback_val_ratio": 0.20,

    # 工况变化判断阈值
    "condition_eps_V": 3,
    "condition_eps_R": 1,

    # 最多使用的场景数。None 表示全部。
    "max_train_scenarios": None,
    "max_val_scenarios": None,

    # --------------------------------------------------------
    # 模仿预训练
    # 教师控制器：du_teacher=clip(u_target-u, ±du_max)
    # --------------------------------------------------------
    "imitation_epochs": 30,
    "imitation_lr": 1e-3,

    # --------------------------------------------------------
    # 多步 DPC 训练
    # --------------------------------------------------------
    "epochs": 400,
    "batch_size": 256,
    "lr": 1e-4,
    "weight_decay": 1e-6,
    "grad_clip": 5.0,
    "early_stop_patience": 100,

    # 逐步增加滚动长度，减少初始训练发散
    # (起始 epoch, horizon)
    "horizon_schedule": [
        [1, 10],       # 2 ms
        [51, 20],       # 2 ms
        [151, 50],       # 2 ms
        [251, 70],       # 2 ms
    ],

    # 多步损失折扣
    "gamma": 0.995,

    # --------------------------------------------------------
    # 损失权重
    # --------------------------------------------------------
    "loss_weights": {
        "voltage": 12.0,
        "angle_target": 1.0,
        "current_limit": 30.0,
        "current_magnitude": 0.02,
        "action": 0.03,
        "action_smooth": 0.08,
        "angle_progress": 0.10,
        "integral": 0.005,
        "out_of_domain": 20.0,

        "terminal_voltage": 40.0,
        "terminal_angle": 6.0,
        "terminal_action": 1.0,
    },

    # 损失归一化尺度
    "loss_scales": {
        "Vout": 5.0,
        "angle": [30.0, 30.0, 20.0],
    },

    # --------------------------------------------------------
    # 其他
    # --------------------------------------------------------
    "seed": 42,
    "num_workers": 0,       # Windows 建议使用 0
    "validation_rollout_count": 20,
    "print_every": 5,
}


POLICY_FEATURE_NAMES = [
    "Vout",
    "Ip_rms",
    "Is_rms",
    "voltage_error",
    "voltage_error_integral",
    "a",
    "b",
    "c",
    "a_target",
    "b_target",
    "c_target",
    "a_error",
    "b_error",
    "c_error",
    "da_prev",
    "db_prev",
    "dc_prev",
    "Vin",
    "Rload",
    "Vref",
]


# ============================================================
# 2. 通用工具
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_torch_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_scaler_file(path: Path):
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e_pickle:
            raise RuntimeError(
                f"无法读取 scaler：{path.resolve()}\n"
                f"joblib.load: {repr(e_joblib)}\n"
                f"pickle.load: {repr(e_pickle)}"
            )


def scaler_mean_scale(scaler, expected_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 sklearn StandardScaler 或 {"mean": ..., "std": ...} 转成 mean/scale。
    """
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.asarray(scaler.mean_, dtype=np.float32).reshape(-1)
        scale = np.asarray(scaler.scale_, dtype=np.float32).reshape(-1)
    elif isinstance(scaler, dict) and "mean" in scaler and "std" in scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(-1)
        scale = np.asarray(scaler["std"], dtype=np.float32).reshape(-1)
    else:
        raise RuntimeError(
            f"无法从 scaler 中提取 mean/scale，类型为：{type(scaler)}"
        )

    if len(mean) != expected_dim or len(scale) != expected_dim:
        raise RuntimeError(
            f"scaler 维度错误：期望 {expected_dim}，"
            f"mean={len(mean)}, scale={len(scale)}"
        )

    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def transform_with_scaler(scaler, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if hasattr(scaler, "transform") and callable(scaler.transform):
        return scaler.transform(x).astype(np.float32)

    if hasattr(scaler, "transform_np") and callable(scaler.transform_np):
        return scaler.transform_np(x).astype(np.float32)

    if isinstance(scaler, dict) and "mean" in scaler and "std" in scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, -1)
        std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, -1)
        std = np.where(np.abs(std) < 1e-12, 1.0, std)
        return ((x - mean) / std).astype(np.float32)

    raise RuntimeError(f"无法识别 scaler 类型：{type(scaler)}")


def natural_key(text: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def extract_state_dict(obj) -> OrderedDict:
    if isinstance(obj, nn.Module):
        return OrderedDict(obj.state_dict())

    if isinstance(obj, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "policy_state_dict",
            "net_state_dict",
        ]:
            if key in obj and isinstance(obj[key], dict):
                return OrderedDict(obj[key])

        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return OrderedDict(obj)

        for value in obj.values():
            if (
                isinstance(value, dict)
                and value
                and all(torch.is_tensor(v) for v in value.values())
            ):
                return OrderedDict(value)

    raise RuntimeError("无法从 checkpoint 中解析 state_dict。")


def infer_linear_dims(state_dict: OrderedDict):
    items = [
        (k, v)
        for k, v in state_dict.items()
        if k.endswith("weight") and torch.is_tensor(v) and v.ndim == 2
    ]
    items = sorted(items, key=lambda kv: natural_key(kv[0]))

    if len(items) < 2:
        raise RuntimeError("state_dict 中没有足够的线性层权重。")

    shapes = [tuple(v.shape) for _, v in items]
    input_dim = shapes[0][1]
    hidden_dims = [shape[0] for shape in shapes[:-1]]
    output_dim = shapes[-1][0]
    return input_dim, hidden_dims, output_dim, items


# ============================================================
# 3. 稳态最优角网络
# ============================================================

class GenericMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h

        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def remap_linear_state_dict(
    state_dict: OrderedDict,
    weight_items,
    dropout: float = 0.0,
) -> OrderedDict:
    """
    将任意命名的纯 MLP 线性层映射到 GenericMLP.net。
    当前稳态网络默认没有 dropout。
    """
    if dropout != 0:
        raise RuntimeError("稳态网络自动重映射目前只支持 dropout=0。")

    new_state = OrderedDict()

    for i, (weight_key, weight_value) in enumerate(weight_items):
        linear_index = 2 * i
        bias_key = weight_key[:-len("weight")] + "bias"

        if bias_key not in state_dict:
            raise RuntimeError(f"找不到偏置参数：{bias_key}")

        new_state[f"net.{linear_index}.weight"] = weight_value
        new_state[f"net.{linear_index}.bias"] = state_dict[bias_key]

    return new_state


class SteadyPolicyWrapper:
    def __init__(self, cfg: dict):
        policy_dir = Path(cfg["steady_policy_dir"])
        pth_path = policy_dir / cfg["steady_policy_pth"]
        scaler_path = policy_dir / cfg["steady_policy_scaler"]
        bounds_path = policy_dir / cfg["steady_policy_angle_bounds"]

        if not pth_path.exists():
            raise FileNotFoundError(f"找不到稳态网络：{pth_path.resolve()}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"找不到稳态 scaler：{scaler_path.resolve()}")

        checkpoint = load_torch_file(pth_path)
        state_dict = extract_state_dict(checkpoint)

        clean_state = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            clean_state[key] = value

        input_dim, hidden_dims, output_dim, weight_items = infer_linear_dims(clean_state)

        self.model = GenericMLP(input_dim, hidden_dims, output_dim)
        remapped = remap_linear_state_dict(clean_state, weight_items)
        self.model.load_state_dict(remapped, strict=True)
        self.model.eval()

        self.scaler = load_scaler_file(scaler_path)
        self.input_dim = input_dim
        self.cfg = cfg

        if bounds_path.exists():
            bounds = np.load(bounds_path).astype(np.float32)
            if bounds.shape == (3, 2):
                self.angle_low = bounds[:, 0]
                self.angle_high = bounds[:, 1]
            elif bounds.shape == (2, 3):
                self.angle_low = bounds[0]
                self.angle_high = bounds[1]
            else:
                raise RuntimeError(
                    f"无法识别稳态角度边界形状：{bounds.shape}"
                )
        else:
            self.angle_low = np.asarray(cfg["angle_low"], dtype=np.float32)
            self.angle_high = np.asarray(cfg["angle_high"], dtype=np.float32)

        print("稳态最优角网络：")
        print(f"  input_dim   = {input_dim}")
        print(f"  hidden_dims = {hidden_dims}")
        print(f"  output_dim  = {output_dim}")

    def _make_input(self, vin: np.ndarray, rload: np.ndarray) -> np.ndarray:
        vin = np.asarray(vin, dtype=np.float32).reshape(-1)
        rload = np.asarray(rload, dtype=np.float32).reshape(-1)

        if self.input_dim == 2:
            x = np.column_stack([vin, rload])
        elif self.input_dim == 3:
            pref = np.full(
                len(vin),
                float(self.cfg["pref_value"]),
                dtype=np.float32,
            )
            x = np.column_stack([vin, rload, pref])
        else:
            raise RuntimeError(
                f"稳态网络输入维度为 {self.input_dim}，"
                "当前代码仅支持 [Vin,Rload] 或 [Vin,Rload,pref]。"
            )

        return x.astype(np.float32)

    def _map_output(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)[:, :3]
        mode = self.cfg["steady_output_mode"]

        if mode == "direct_angles":
            angles = raw
        elif mode == "normalized01_to_bounds":
            y = np.clip(raw, 0.0, 1.0)
            angles = self.angle_low + y * (self.angle_high - self.angle_low)
        elif mode == "sigmoid_to_bounds":
            y = 1.0 / (1.0 + np.exp(-raw))
            angles = self.angle_low + y * (self.angle_high - self.angle_low)
        elif mode == "auto":
            in_angle = np.mean(
                (raw >= self.angle_low - 5.0)
                & (raw <= self.angle_high + 5.0)
            )
            in_01 = np.mean((raw >= -0.05) & (raw <= 1.05))

            if in_angle > 0.95 and np.mean(raw) > 10.0:
                angles = raw
            elif in_01 > 0.95:
                y = np.clip(raw, 0.0, 1.0)
                angles = self.angle_low + y * (
                    self.angle_high - self.angle_low
                )
            else:
                y = 1.0 / (1.0 + np.exp(-raw))
                angles = self.angle_low + y * (
                    self.angle_high - self.angle_low
                )
        else:
            raise ValueError(f"未知 steady_output_mode：{mode}")

        return np.clip(
            angles,
            self.angle_low.reshape(1, -1),
            self.angle_high.reshape(1, -1),
        ).astype(np.float32)

    @torch.no_grad()
    def predict_batch(
        self,
        vin: np.ndarray,
        rload: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        x = self._make_input(vin, rload)
        x_scaled = transform_with_scaler(self.scaler, x)

        outputs = []
        for start in range(0, len(x_scaled), batch_size):
            xb = torch.from_numpy(x_scaled[start:start + batch_size])
            outputs.append(self.model(xb).cpu().numpy())

        raw = np.vstack(outputs)
        return self._map_output(raw)


# ============================================================
# 4. 动态代理模型
# ============================================================

class DynamicSurrogateMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h

        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrozenDynamicModel(nn.Module):
    """
    将动态代理模型和输入/输出标准化封装为可微模块。
    参数被冻结，但梯度仍可从输出传回策略动作和状态。
    """

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()

        model_dir = Path(cfg["dynamic_model_dir"])
        pth_path = model_dir / cfg["dynamic_model_pth"]
        x_scaler_path = model_dir / cfg["dynamic_x_scaler"]
        y_scaler_path = model_dir / cfg["dynamic_y_scaler"]

        if not pth_path.exists():
            raise FileNotFoundError(f"找不到动态代理模型：{pth_path.resolve()}")

        checkpoint = load_torch_file(pth_path)

        hidden_dims = checkpoint.get(
            "hidden_dims",
            checkpoint.get("config", {}).get(
                "hidden_dims",
                [256, 256, 128, 64],
            ),
        )
        input_dim = int(checkpoint.get("input_dim", 11))
        output_dim = int(checkpoint.get("output_dim", 3))
        dropout = float(
            checkpoint.get("config", {}).get("dropout", 0.0)
        )

        self.model = DynamicSurrogateMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        x_scaler = load_scaler_file(x_scaler_path)
        y_scaler = load_scaler_file(y_scaler_path)

        x_mean, x_scale = scaler_mean_scale(x_scaler, input_dim)
        y_mean, y_scale = scaler_mean_scale(y_scaler, output_dim)

        self.register_buffer(
            "x_mean",
            torch.as_tensor(x_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "x_scale",
            torch.as_tensor(x_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "y_mean",
            torch.as_tensor(y_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "y_scale",
            torch.as_tensor(y_scale, dtype=torch.float32),
        )

        self.to(device)

        print("动态代理模型：")
        print(f"  input_dim   = {input_dim}")
        print(f"  hidden_dims = {hidden_dims}")
        print(f"  output_dim  = {output_dim}")

    def predict_delta(
        self,
        state: torch.Tensor,
        angles: torch.Tensor,
        delta_u: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        state:     [B,3] = [Vout, Ip, Is]
        angles:    [B,3] = [a,b,c]，动作施加前角度
        delta_u:   [B,3] = 实际施加的角度增量
        condition: [B,2] = [Vin_new,Rload_new]
        """
        raw_input = torch.cat(
            [state, angles, delta_u, condition],
            dim=1,
        )
        scaled_input = (raw_input - self.x_mean) / self.x_scale
        scaled_delta = self.model(scaled_input)
        delta = scaled_delta * self.y_scale + self.y_mean
        return delta


# ============================================================
# 5. 场景数据
# ============================================================

REQUIRED_COLUMNS = [
    "time",
    "Vin",
    "Rload",
    "a",
    "b",
    "c",
    "Vout_avg",
    "Ip_rms",
    "Is_rms",
]


def condition_in_filter(vin: float, rload: float, cfg: dict) -> bool:
    """判断工况是否位于动态策略训练范围内。"""
    if not cfg.get("condition_filter_enabled", False):
        return True
    return (
        float(cfg["filter_vin_min"]) <= float(vin) <= float(cfg["filter_vin_max"])
        and
        float(cfg["filter_rload_min"]) <= float(rload) <= float(cfg["filter_rload_max"])
    )


def find_all_csv_files(data_dirs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    base = Path.cwd()

    for directory in data_dirs:
        path = base / directory
        if not path.exists():
            print(f"警告：数据目录不存在：{path.resolve()}")
            continue
        files.extend(sorted(path.glob("*.csv")))

    # 去重
    unique = {}
    for file in files:
        unique[str(file.resolve())] = file

    return sorted(unique.values(), key=lambda p: str(p))


def read_trajectory(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列：{missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.sort_values("time")
    df = df.drop_duplicates("time", keep="last").reset_index(drop=True)

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    return df


def resolve_split_paths(
    paths: Sequence[str],
    all_files: Sequence[Path],
) -> List[Path]:
    """
    split_info.json 中可能保存绝对路径。路径失效时按文件名匹配。
    """
    by_name: Dict[str, List[Path]] = {}
    for file in all_files:
        by_name.setdefault(file.name, []).append(file)

    resolved: List[Path] = []

    for raw in paths:
        path = Path(raw)
        if path.exists():
            resolved.append(path)
            continue

        candidates = by_name.get(path.name, [])
        if candidates:
            resolved.append(candidates[0])

    # 去重
    unique = {}
    for file in resolved:
        unique[str(file.resolve())] = file

    return list(unique.values())


def split_files(
    all_files: Sequence[Path],
    cfg: dict,
) -> Tuple[List[Path], List[Path], List[Path]]:
    split_path = (
        Path(cfg["dynamic_model_dir"])
        / cfg["dynamic_split_info"]
    )

    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as file:
            split_info = json.load(file)

        train_files = resolve_split_paths(
            split_info.get("train_files", []),
            all_files,
        )
        val_files = resolve_split_paths(
            split_info.get("val_files", []),
            all_files,
        )
        test_files = resolve_split_paths(
            split_info.get("test_files", []),
            all_files,
        )

        if train_files and val_files:
            print("使用动态代理模型 split_info.json 中的轨迹划分。")
            return train_files, val_files, test_files

    print("未找到有效 split_info.json，重新按轨迹文件划分。")
    files = list(all_files)
    random.shuffle(files)

    n_total = len(files)
    n_train = max(
        1,
        int(round(n_total * float(cfg["fallback_train_ratio"]))),
    )
    n_val = max(
        1,
        int(round(n_total * float(cfg["fallback_val_ratio"]))),
    )

    if n_train + n_val > n_total:
        n_val = max(1, n_total - n_train)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def extract_scenario_candidates(
    files: Sequence[Path],
    cfg: dict,
) -> Tuple[List[dict], List[dict], pd.DataFrame]:
    """
    仅保留 Vin/Rload 位于指定范围内的动态策略训练场景。
    构造 k->k+1 场景时，要求当前点和下一点都在范围内，
    避免跨边界样本进入训练。
    """
    step_scenarios: List[dict] = []
    hold_candidates: List[dict] = []
    state_rows: List[dict] = []

    eps_v = float(cfg["condition_eps_V"])
    eps_r = float(cfg["condition_eps_R"])

    total_pairs = 0
    kept_pairs = 0
    skipped_pairs = 0

    for path in files:
        try:
            df = read_trajectory(path)
        except Exception as exc:
            print(f"跳过 {path}: {exc}")
            continue

        if len(df) < 2:
            continue

        # 只统计范围内状态，用于后续状态域和电流限值估计
        for _, row in df.iterrows():
            vin_i = float(row["Vin"])
            rload_i = float(row["Rload"])

            if not condition_in_filter(vin_i, rload_i, cfg):
                continue

            state_rows.append({
                "Vout": float(row["Vout_avg"]),
                "Ip": float(row["Ip_rms"]),
                "Is": float(row["Is_rms"]),
            })

        vin = df["Vin"].to_numpy(dtype=np.float32)
        rload = df["Rload"].to_numpy(dtype=np.float32)
        dvin = np.diff(vin)
        dr = np.diff(rload)

        for k in range(len(df) - 1):
            total_pairs += 1

            row_k = df.iloc[k]
            row_next = df.iloc[k + 1]

            vin_k = float(row_k["Vin"])
            rload_k = float(row_k["Rload"])
            vin_next = float(row_next["Vin"])
            rload_next = float(row_next["Rload"])

            # 当前点和下一点都必须位于过滤范围内
            if not (
                condition_in_filter(vin_k, rload_k, cfg)
                and condition_in_filter(vin_next, rload_next, cfg)
            ):
                skipped_pairs += 1
                continue

            kept_pairs += 1

            scenario = {
                "source_file": str(path),
                "source_index": int(k),
                "x0": [
                    float(row_k["Vout_avg"]),
                    float(row_k["Ip_rms"]),
                    float(row_k["Is_rms"]),
                ],
                "u0": [
                    float(row_k["a"]),
                    float(row_k["b"]),
                    float(row_k["c"]),
                ],
                "condition_new": [
                    vin_next,
                    rload_next,
                ],
            }

            changed = (
                abs(float(dvin[k])) > eps_v
                or abs(float(dr[k])) > eps_r
            )

            if changed:
                scenario["scenario_type"] = "condition_step"
                step_scenarios.append(scenario)
            else:
                scenario["scenario_type"] = "condition_hold"
                hold_candidates.append(scenario)

    if cfg.get("condition_filter_enabled", False):
        print(
            f"工况过滤范围: Vin=[{cfg['filter_vin_min']}, {cfg['filter_vin_max']}] V, "
            f"Rload=[{cfg['filter_rload_min']}, {cfg['filter_rload_max']}] Ω"
        )
        print(
            f"相邻样本对: 总数={total_pairs}, 保留={kept_pairs}, 过滤={skipped_pairs}"
        )
        print(
            f"保留状态样本={len(state_rows)}, "
            f"阶跃场景={len(step_scenarios)}, 保持候选={len(hold_candidates)}"
        )

    return step_scenarios, hold_candidates, pd.DataFrame(state_rows)

def build_scenarios(
    files: Sequence[Path],
    steady_policy: SteadyPolicyWrapper,
    cfg: dict,
    max_scenarios: Optional[int],
    random_seed_offset: int,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    step_scenarios, hold_candidates, states_df = extract_scenario_candidates(
        files,
        cfg,
    )

    rng = random.Random(int(cfg["seed"]) + random_seed_offset)

    if not step_scenarios:
        raise RuntimeError(
            "没有提取到工况阶跃场景。请确认轨迹中 Vin/Rload 每 20 ms 发生变化。"
        )

    hold_ratio = float(cfg["hold_scenario_ratio"])
    if not 0.0 <= hold_ratio < 1.0:
        raise ValueError("hold_scenario_ratio 必须在 [0,1) 内。")

    n_hold = int(round(
        len(step_scenarios) * hold_ratio / max(1.0 - hold_ratio, 1e-6)
    ))
    n_hold = min(n_hold, len(hold_candidates))

    selected_hold = (
        rng.sample(hold_candidates, n_hold)
        if n_hold > 0
        else []
    )

    scenarios = step_scenarios + selected_hold
    rng.shuffle(scenarios)

    if max_scenarios is not None and len(scenarios) > int(max_scenarios):
        scenarios = rng.sample(scenarios, int(max_scenarios))

    x0 = np.asarray([s["x0"] for s in scenarios], dtype=np.float32)
    u0 = np.asarray([s["u0"] for s in scenarios], dtype=np.float32)
    condition = np.asarray(
        [s["condition_new"] for s in scenarios],
        dtype=np.float32,
    )

    target_u = steady_policy.predict_batch(
        condition[:, 0],
        condition[:, 1],
    )

    scenario_type = np.asarray(
        [s["scenario_type"] for s in scenarios],
        dtype=object,
    )
    source_file = np.asarray(
        [s["source_file"] for s in scenarios],
        dtype=object,
    )
    source_index = np.asarray(
        [s["source_index"] for s in scenarios],
        dtype=np.int64,
    )

    arrays = {
        "x0": x0,
        "u0": u0,
        "condition": condition,
        "target_u": target_u.astype(np.float32),
        "scenario_type": scenario_type,
        "source_file": source_file,
        "source_index": source_index,
    }

    print(
        f"场景数量={len(scenarios)}，"
        f"工况阶跃={len(step_scenarios)}，"
        f"加入保持场景={len(selected_hold)}"
    )

    return arrays, states_df


class ScenarioDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.x0 = torch.as_tensor(arrays["x0"], dtype=torch.float32)
        self.u0 = torch.as_tensor(arrays["u0"], dtype=torch.float32)
        self.condition = torch.as_tensor(
            arrays["condition"],
            dtype=torch.float32,
        )
        self.target_u = torch.as_tensor(
            arrays["target_u"],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, index: int):
        return (
            self.x0[index],
            self.u0[index],
            self.condition[index],
            self.target_u[index],
        )


# ============================================================
# 6. 动态策略网络
# ============================================================

def build_policy_input(
    state: torch.Tensor,
    integral_state: torch.Tensor,
    angles: torch.Tensor,
    target_angles: torch.Tensor,
    previous_du: torch.Tensor,
    condition: torch.Tensor,
    vref: torch.Tensor,
) -> torch.Tensor:
    voltage_error = vref - state[:, 0:1]
    angle_error = target_angles - angles

    return torch.cat(
        [
            state,
            voltage_error,
            integral_state,
            angles,
            target_angles,
            angle_error,
            previous_du,
            condition,
            vref,
        ],
        dim=1,
    )


def make_policy_normalizer(
    train_arrays: Dict[str, np.ndarray],
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    x0 = train_arrays["x0"]
    u0 = train_arrays["u0"]
    target_u = train_arrays["target_u"]
    condition = train_arrays["condition"]

    n = len(x0)
    vref = np.full((n, 1), float(cfg["Vref"]), dtype=np.float32)
    voltage_error = vref - x0[:, 0:1]
    integral = np.zeros((n, 1), dtype=np.float32)
    angle_error = target_u - u0
    previous_du = np.zeros((n, 3), dtype=np.float32)

    raw = np.concatenate(
        [
            x0,
            voltage_error,
            integral,
            u0,
            target_u,
            angle_error,
            previous_du,
            condition,
            vref,
        ],
        axis=1,
    )

    mean = raw.mean(axis=0).astype(np.float32)
    std = raw.std(axis=0).astype(np.float32)

    # 对初始场景中恒为零或变化很小的变量，给出物理合理尺度。
    minimum_std = np.asarray(
        [
            5.0,   # Vout
            1.0,   # Ip
            1.0,   # Is
            5.0,   # voltage error
            0.05,  # integral
            15.0, 15.0, 10.0,  # current angles
            15.0, 15.0, 10.0,  # target angles
            5.0, 5.0, 5.0,     # angle errors
            1.0, 1.0, 1.0,     # previous du
            20.0, 10.0,         # Vin, Rload
            5.0,                # Vref
        ],
        dtype=np.float32,
    )

    std = np.maximum(std, minimum_std)

    # 积分量、上一步动作、Vref 初始都可能恒定，均值固定为零/给定更直观。
    mean[4] = 0.0
    mean[14:17] = 0.0
    mean[19] = float(cfg["Vref"])

    return mean, std


class DynamicPolicyMLP(nn.Module):
    def __init__(
        self,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        hidden_dims: Sequence[int],
        du_max: Sequence[float],
        dropout: float = 0.0,
    ):
        super().__init__()

        input_dim = len(input_mean)
        layers: List[nn.Module] = []
        last = input_dim

        for hidden in hidden_dims:
            layers.append(nn.Linear(last, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = hidden

        layers.append(nn.Linear(last, 3))
        self.net = nn.Sequential(*layers)

        self.register_buffer(
            "input_mean",
            torch.as_tensor(input_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "input_std",
            torch.as_tensor(input_std, dtype=torch.float32),
        )
        self.register_buffer(
            "du_max",
            torch.as_tensor(du_max, dtype=torch.float32),
        )

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        normalized = (raw_input - self.input_mean) / self.input_std
        raw_action = self.net(normalized)
        return self.du_max * torch.tanh(raw_action)


# ============================================================
# 7. 多步滚动与损失
# ============================================================

def horizon_for_epoch(epoch: int, schedule: Sequence[Sequence[int]]) -> int:
    horizon = int(schedule[0][1])
    for start_epoch, current_horizon in schedule:
        if epoch >= int(start_epoch):
            horizon = int(current_horizon)
    return horizon


def derive_limits_and_domain(
    states_df: pd.DataFrame,
    cfg: dict,
) -> dict:
    if states_df.empty:
        raise RuntimeError("无法从训练数据统计状态范围。")

    q = float(cfg["current_limit_quantile"])
    margin = float(cfg["current_limit_margin"])

    if cfg["Ip_limit"] is None:
        ip_limit = float(states_df["Ip"].quantile(q) * margin)
    else:
        ip_limit = float(cfg["Ip_limit"])

    if cfg["Is_limit"] is None:
        is_limit = float(states_df["Is"].quantile(q) * margin)
    else:
        is_limit = float(cfg["Is_limit"])

    lower = np.asarray(
        [
            max(0.0, float(states_df["Vout"].min()) * 0.50),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    upper = np.asarray(
        [
            max(
                float(states_df["Vout"].max()) * 1.50,
                float(cfg["Vref"]) * 1.50,
            ),
            max(float(states_df["Ip"].max()) * 1.50, ip_limit * 1.20),
            max(float(states_df["Is"].max()) * 1.50, is_limit * 1.20),
        ],
        dtype=np.float32,
    )

    return {
        "Ip_limit": ip_limit,
        "Is_limit": is_limit,
        "state_lower": lower,
        "state_upper": upper,
    }


def rollout_loss(
    policy: DynamicPolicyMLP,
    dynamic_model: FrozenDynamicModel,
    batch,
    cfg: dict,
    runtime: dict,
    horizon: int,
    collect: bool = False,
):
    x0, u0, condition, target_u = batch
    device = next(policy.parameters()).device

    state = x0.to(device)
    angles = u0.to(device)
    condition = condition.to(device)
    target_u = target_u.to(device)

    batch_size = state.shape[0]

    vref = torch.full(
        (batch_size, 1),
        float(cfg["Vref"]),
        dtype=torch.float32,
        device=device,
    )
    integral = torch.zeros(
        (batch_size, 1),
        dtype=torch.float32,
        device=device,
    )
    previous_du = torch.zeros(
        (batch_size, 3),
        dtype=torch.float32,
        device=device,
    )

    angle_low = torch.as_tensor(
        cfg["angle_low"],
        dtype=torch.float32,
        device=device,
    )
    angle_high = torch.as_tensor(
        cfg["angle_high"],
        dtype=torch.float32,
        device=device,
    )
    angle_scale = torch.as_tensor(
        cfg["loss_scales"]["angle"],
        dtype=torch.float32,
        device=device,
    )
    du_max = torch.as_tensor(
        cfg["du_max"],
        dtype=torch.float32,
        device=device,
    )

    state_lower = torch.as_tensor(
        runtime["state_lower"],
        dtype=torch.float32,
        device=device,
    )
    state_upper = torch.as_tensor(
        runtime["state_upper"],
        dtype=torch.float32,
        device=device,
    )

    ip_limit = float(runtime["Ip_limit"])
    is_limit = float(runtime["Is_limit"])
    v_scale = float(cfg["loss_scales"]["Vout"])
    gamma = float(cfg["gamma"])
    weights = cfg["loss_weights"]

    total_loss = torch.zeros((), dtype=torch.float32, device=device)

    initial_angle_error_norm = torch.linalg.vector_norm(
        (target_u - angles) / angle_scale,
        dim=1,
    )

    last_du = previous_du
    trace = []

    for step in range(horizon):
        policy_input = build_policy_input(
            state=state,
            integral_state=integral,
            angles=angles,
            target_angles=target_u,
            previous_du=previous_du,
            condition=condition,
            vref=vref,
        )

        commanded_du = policy(policy_input)

        next_angles = torch.minimum(
            torch.maximum(angles + commanded_du, angle_low),
            angle_high,
        )
        effective_du = next_angles - angles

        predicted_delta = dynamic_model.predict_delta(
            state=state,
            angles=angles,
            delta_u=effective_du,
            condition=condition,
        )
        raw_next_state = state + predicted_delta

        # 代理模型适用域惩罚
        below = torch.relu(state_lower - raw_next_state)
        above = torch.relu(raw_next_state - state_upper)
        out_of_domain_penalty = (
            (below / torch.clamp(state_upper - state_lower, min=1e-6)) ** 2
            + (above / torch.clamp(state_upper - state_lower, min=1e-6)) ** 2
        ).mean()

        # 为防止滚动数值爆炸，使用宽松物理范围裁剪。
        next_state = torch.minimum(
            torch.maximum(raw_next_state, state_lower),
            state_upper,
        )

        voltage_error_next = vref - next_state[:, 0:1]
        angle_error_next = target_u - next_angles

        normalized_voltage_error = voltage_error_next / v_scale
        normalized_angle_error = angle_error_next / angle_scale

        ip_violation = torch.relu(
            (next_state[:, 1] - ip_limit) / max(ip_limit, 1e-6)
        )
        is_violation = torch.relu(
            (next_state[:, 2] - is_limit) / max(is_limit, 1e-6)
        )

        current_limit_loss = (
            ip_violation.pow(2).mean()
            + is_violation.pow(2).mean()
        )

        current_magnitude_loss = (
            (next_state[:, 1] / max(ip_limit, 1e-6)).pow(2).mean()
            + (next_state[:, 2] / max(is_limit, 1e-6)).pow(2).mean()
        )

        action_loss = (effective_du / du_max).pow(2).mean()
        action_smooth_loss = (
            (effective_du - previous_du) / du_max
        ).pow(2).mean()

        current_angle_error_norm = torch.linalg.vector_norm(
            normalized_angle_error,
            dim=1,
        )
        angle_progress_loss = torch.relu(
            current_angle_error_norm - initial_angle_error_norm
        ).pow(2).mean()

        step_loss = (
            float(weights["voltage"])
            * normalized_voltage_error.pow(2).mean()
            + float(weights["angle_target"])
            * normalized_angle_error.pow(2).mean()
            + float(weights["current_limit"])
            * current_limit_loss
            + float(weights["current_magnitude"])
            * current_magnitude_loss
            + float(weights["action"])
            * action_loss
            + float(weights["action_smooth"])
            * action_smooth_loss
            + float(weights["angle_progress"])
            * angle_progress_loss
            + float(weights["integral"])
            * integral.pow(2).mean()
            + float(weights["out_of_domain"])
            * out_of_domain_penalty
        )

        total_loss = total_loss + (gamma ** step) * step_loss

        # 更新积分状态。这里使用 next-state 电压误差。
        integral = torch.clamp(
            integral
            + float(cfg["Ts_ctrl"]) * voltage_error_next,
            min=float(cfg["integral_min"]),
            max=float(cfg["integral_max"]),
        )

        previous_du = effective_du
        last_du = effective_du
        state = next_state
        angles = next_angles

        if collect:
            trace.append({
                "state": state.detach().cpu(),
                "angles": angles.detach().cpu(),
                "du": effective_du.detach().cpu(),
                "integral": integral.detach().cpu(),
            })

    terminal_voltage_error = (vref - state[:, 0:1]) / v_scale
    terminal_angle_error = (target_u - angles) / angle_scale

    terminal_loss = (
        float(weights["terminal_voltage"])
        * terminal_voltage_error.pow(2).mean()
        + float(weights["terminal_angle"])
        * terminal_angle_error.pow(2).mean()
        + float(weights["terminal_action"])
        * (last_du / du_max).pow(2).mean()
    )

    total_loss = total_loss / max(horizon, 1) + terminal_loss

    metrics = {
        "loss": float(total_loss.detach().cpu()),
        "final_voltage_mae": float(
            torch.mean(torch.abs(vref[:, 0] - state[:, 0]))
            .detach()
            .cpu()
        ),
        "final_angle_mae": float(
            torch.mean(torch.abs(target_u - angles))
            .detach()
            .cpu()
        ),
        "max_Ip": float(torch.max(state[:, 1]).detach().cpu()),
        "max_Is": float(torch.max(state[:, 2]).detach().cpu()),
    }

    return total_loss, metrics, trace


# ============================================================
# 8. 训练
# ============================================================

def imitation_pretrain(
    policy: DynamicPolicyMLP,
    train_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> List[dict]:
    epochs = int(cfg["imitation_epochs"])
    if epochs <= 0:
        return []

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=float(cfg["imitation_lr"]),
    )
    loss_fn = nn.MSELoss()
    history = []

    print("\n开始模仿预训练……")

    for epoch in range(1, epochs + 1):
        policy.train()
        loss_sum = 0.0
        count = 0

        for x0, u0, condition, target_u in train_loader:
            x0 = x0.to(device)
            u0 = u0.to(device)
            condition = condition.to(device)
            target_u = target_u.to(device)

            batch_size = x0.shape[0]
            vref = torch.full(
                (batch_size, 1),
                float(cfg["Vref"]),
                device=device,
            )
            integral = torch.zeros(
                (batch_size, 1),
                device=device,
            )
            previous_du = torch.zeros(
                (batch_size, 3),
                device=device,
            )

            raw_input = build_policy_input(
                state=x0,
                integral_state=integral,
                angles=u0,
                target_angles=target_u,
                previous_du=previous_du,
                condition=condition,
                vref=vref,
            )

            predicted_du = policy(raw_input)

            du_max = torch.as_tensor(
                cfg["du_max"],
                dtype=torch.float32,
                device=device,
            )
            teacher_du = torch.minimum(
                torch.maximum(target_u - u0, -du_max),
                du_max,
            )

            loss = loss_fn(predicted_du, teacher_du)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                float(cfg["grad_clip"]),
            )
            optimizer.step()

            loss_sum += float(loss.detach().cpu()) * batch_size
            count += batch_size

        mean_loss = loss_sum / max(count, 1)
        history.append({
            "stage": "imitation",
            "epoch": epoch,
            "loss": mean_loss,
        })

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[Imitation] epoch={epoch:03d} "
                f"loss={mean_loss:.6e}"
            )

    return history


@torch.no_grad()
def evaluate_policy(
    policy: DynamicPolicyMLP,
    dynamic_model: FrozenDynamicModel,
    loader: DataLoader,
    cfg: dict,
    runtime: dict,
    horizon: int,
) -> dict:
    policy.eval()

    accum = {
        "loss": 0.0,
        "final_voltage_mae": 0.0,
        "final_angle_mae": 0.0,
        "max_Ip": 0.0,
        "max_Is": 0.0,
    }
    count = 0

    for batch in loader:
        batch_size = batch[0].shape[0]

        _, metrics, _ = rollout_loss(
            policy,
            dynamic_model,
            batch,
            cfg,
            runtime,
            horizon,
            collect=False,
        )

        accum["loss"] += metrics["loss"] * batch_size
        accum["final_voltage_mae"] += (
            metrics["final_voltage_mae"] * batch_size
        )
        accum["final_angle_mae"] += (
            metrics["final_angle_mae"] * batch_size
        )
        accum["max_Ip"] = max(accum["max_Ip"], metrics["max_Ip"])
        accum["max_Is"] = max(accum["max_Is"], metrics["max_Is"])
        count += batch_size

    if count == 0:
        raise RuntimeError("验证集为空。")

    accum["loss"] /= count
    accum["final_voltage_mae"] /= count
    accum["final_angle_mae"] /= count
    return accum


def save_checkpoint(
    path: Path,
    policy: DynamicPolicyMLP,
    cfg: dict,
    runtime: dict,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    epoch: int,
    val_metrics: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "input_dim": len(POLICY_FEATURE_NAMES),
            "output_dim": 3,
            "hidden_dims": cfg["policy_hidden_dims"],
            "dropout": cfg["policy_dropout"],
            "input_names": POLICY_FEATURE_NAMES,
            "output_names": ["da", "db", "dc"],
            "input_mean": input_mean,
            "input_std": input_std,
            "du_max": np.asarray(cfg["du_max"], dtype=np.float32),
            "angle_low": np.asarray(
                cfg["angle_low"],
                dtype=np.float32,
            ),
            "angle_high": np.asarray(
                cfg["angle_high"],
                dtype=np.float32,
            ),
            "Vref": float(cfg["Vref"]),
            "Ts_ctrl": float(cfg["Ts_ctrl"]),
            "runtime": runtime,
            "config": cfg,
            "best_epoch": int(epoch),
            "val_metrics": val_metrics,
        },
        path,
    )


def train_dpc(
    policy: DynamicPolicyMLP,
    dynamic_model: FrozenDynamicModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    runtime: dict,
    output_dir: Path,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    device: torch.device,
) -> List[dict]:
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    best_val = float("inf")
    best_epoch = -1
    wait = 0
    history: List[dict] = []
    checkpoint_path = output_dir / "dynamic_policy_best.pth"

    print("\n开始多步 DPC 训练……")

    for epoch in range(1, int(cfg["epochs"]) + 1):
        horizon = horizon_for_epoch(
            epoch,
            cfg["horizon_schedule"],
        )

        policy.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            optimizer.zero_grad()

            loss, _, _ = rollout_loss(
                policy,
                dynamic_model,
                batch,
                cfg,
                runtime,
                horizon,
                collect=False,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "训练损失出现 NaN/Inf。建议先缩短 horizon、"
                    "检查动态代理模型滚动预测，或降低学习率。"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                float(cfg["grad_clip"]),
            )
            optimizer.step()

            batch_size = batch[0].shape[0]
            train_loss_sum += float(loss.detach().cpu()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)

        val_metrics = evaluate_policy(
            policy,
            dynamic_model,
            val_loader,
            cfg,
            runtime,
            horizon,
        )

        record = {
            "stage": "dpc",
            "epoch": epoch,
            "horizon": horizon,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_final_voltage_mae": val_metrics["final_voltage_mae"],
            "val_final_angle_mae": val_metrics["final_angle_mae"],
            "val_max_Ip": val_metrics["max_Ip"],
            "val_max_Is": val_metrics["max_Is"],
        }
        history.append(record)

        if epoch == 1 or epoch % int(cfg["print_every"]) == 0:
            print(
                f"[DPC] epoch={epoch:03d} H={horizon:02d} "
                f"train={train_loss:.6e} "
                f"val={val_metrics['loss']:.6e} "
                f"Verr={val_metrics['final_voltage_mae']:.4f} "
                f"Aerr={val_metrics['final_angle_mae']:.4f}"
            )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            wait = 0

            save_checkpoint(
                checkpoint_path,
                policy,
                cfg,
                runtime,
                input_mean,
                input_std,
                epoch,
                val_metrics,
            )
        else:
            wait += 1

        if wait >= int(cfg["early_stop_patience"]):
            print(
                f"Early stopping：epoch={epoch}，"
                f"best_epoch={best_epoch}，best_val={best_val:.6e}"
            )
            break

    print(
        f"DPC 训练结束：best_epoch={best_epoch}，"
        f"best_val={best_val:.6e}"
    )
    return history


# ============================================================
# 9. 保存验证滚动轨迹
# ============================================================

@torch.no_grad()
def save_validation_rollouts(
    policy: DynamicPolicyMLP,
    dynamic_model: FrozenDynamicModel,
    val_arrays: Dict[str, np.ndarray],
    cfg: dict,
    runtime: dict,
    output_dir: Path,
    device: torch.device,
) -> None:
    n_save = min(
        int(cfg["validation_rollout_count"]),
        len(val_arrays["x0"]),
    )
    if n_save <= 0:
        return

    final_horizon = horizon_for_epoch(
        int(cfg["epochs"]),
        cfg["horizon_schedule"],
    )

    rows = []
    summaries = []

    policy.eval()

    for scenario_id in range(n_save):
        x0 = torch.as_tensor(
            val_arrays["x0"][scenario_id:scenario_id + 1],
            dtype=torch.float32,
        )
        u0 = torch.as_tensor(
            val_arrays["u0"][scenario_id:scenario_id + 1],
            dtype=torch.float32,
        )
        condition = torch.as_tensor(
            val_arrays["condition"][scenario_id:scenario_id + 1],
            dtype=torch.float32,
        )
        target_u = torch.as_tensor(
            val_arrays["target_u"][scenario_id:scenario_id + 1],
            dtype=torch.float32,
        )

        batch = (x0, u0, condition, target_u)

        _, metrics, trace = rollout_loss(
            policy,
            dynamic_model,
            batch,
            cfg,
            runtime,
            final_horizon,
            collect=True,
        )

        # step=0
        rows.append({
            "scenario_id": scenario_id,
            "step": 0,
            "time_ms": 0.0,
            "Vout": float(x0[0, 0]),
            "Ip_rms": float(x0[0, 1]),
            "Is_rms": float(x0[0, 2]),
            "a": float(u0[0, 0]),
            "b": float(u0[0, 1]),
            "c": float(u0[0, 2]),
            "da": 0.0,
            "db": 0.0,
            "dc": 0.0,
            "Vin": float(condition[0, 0]),
            "Rload": float(condition[0, 1]),
            "a_target": float(target_u[0, 0]),
            "b_target": float(target_u[0, 1]),
            "c_target": float(target_u[0, 2]),
        })

        for step, item in enumerate(trace, start=1):
            state = item["state"][0].numpy()
            angles = item["angles"][0].numpy()
            du = item["du"][0].numpy()

            rows.append({
                "scenario_id": scenario_id,
                "step": step,
                "time_ms": step * float(cfg["Ts_ctrl"]) * 1000.0,
                "Vout": float(state[0]),
                "Ip_rms": float(state[1]),
                "Is_rms": float(state[2]),
                "a": float(angles[0]),
                "b": float(angles[1]),
                "c": float(angles[2]),
                "da": float(du[0]),
                "db": float(du[1]),
                "dc": float(du[2]),
                "Vin": float(condition[0, 0]),
                "Rload": float(condition[0, 1]),
                "a_target": float(target_u[0, 0]),
                "b_target": float(target_u[0, 1]),
                "c_target": float(target_u[0, 2]),
            })

        summaries.append({
            "scenario_id": scenario_id,
            **metrics,
        })

    pd.DataFrame(rows).to_csv(
        output_dir / "validation_rollouts.csv",
        index=False,
    )
    pd.DataFrame(summaries).to_csv(
        output_dir / "validation_rollout_summary.csv",
        index=False,
    )


# ============================================================
# 10. 主程序
# ============================================================

def main() -> None:
    cfg = CONFIG
    set_seed(int(cfg["seed"]))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device = {device}")

    # --------------------------------------------------------
    # 读取轨迹并按文件划分
    # --------------------------------------------------------
    all_files = find_all_csv_files(cfg["data_dirs"])

    if cfg.get("condition_filter_enabled", False):
        print(
            f"动态策略训练工况: Vin=[{cfg['filter_vin_min']}, {cfg['filter_vin_max']}] V, "
            f"Rload=[{cfg['filter_rload_min']}, {cfg['filter_rload_max']}] Ω"
        )
    if len(all_files) < 3:
        raise RuntimeError(
            f"找到的 CSV 文件过少：{len(all_files)}。"
            "请检查 data_dirs。"
        )

    train_files, val_files, test_files = split_files(
        all_files,
        cfg,
    )

    print(
        f"轨迹文件：train={len(train_files)}, "
        f"val={len(val_files)}, test={len(test_files)}"
    )

    # --------------------------------------------------------
    # 加载稳态网络并生成训练场景目标角
    # --------------------------------------------------------
    steady_policy = SteadyPolicyWrapper(cfg)

    train_arrays, train_states_df = build_scenarios(
        train_files,
        steady_policy,
        cfg,
        cfg["max_train_scenarios"],
        random_seed_offset=0,
    )
    val_arrays, val_states_df = build_scenarios(
        val_files,
        steady_policy,
        cfg,
        cfg["max_val_scenarios"],
        random_seed_offset=10000,
    )

    train_dataset = ScenarioDataset(train_arrays)
    val_dataset = ScenarioDataset(val_arrays)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=int(cfg["num_workers"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg["num_workers"]),
    )

    # --------------------------------------------------------
    # 动态代理模型与运行约束
    # --------------------------------------------------------
    dynamic_model = FrozenDynamicModel(cfg, device)

    combined_states_df = pd.concat(
        [train_states_df, val_states_df],
        ignore_index=True,
    )
    runtime = derive_limits_and_domain(
        combined_states_df,
        cfg,
    )

    print("运行约束：")
    print(f"  Ip_limit = {runtime['Ip_limit']:.6f}")
    print(f"  Is_limit = {runtime['Is_limit']:.6f}")
    print(f"  state_lower = {runtime['state_lower']}")
    print(f"  state_upper = {runtime['state_upper']}")

    # --------------------------------------------------------
    # 策略网络
    # --------------------------------------------------------
    input_mean, input_std = make_policy_normalizer(
        train_arrays,
        cfg,
    )

    policy = DynamicPolicyMLP(
        input_mean=input_mean,
        input_std=input_std,
        hidden_dims=cfg["policy_hidden_dims"],
        du_max=cfg["du_max"],
        dropout=float(cfg["policy_dropout"]),
    ).to(device)

    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"小型动态策略网络: 20 -> {' -> '.join(map(str, cfg['policy_hidden_dims']))} -> 3")
    print(f"可训练参数量: {trainable_params:,}; 每步相角增量限幅: {cfg['du_max']} deg")

    # 保存输入归一化参数，便于后续 DSP/Python 推理
    joblib.dump(
        {
            "mean": input_mean,
            "std": input_std,
            "columns": POLICY_FEATURE_NAMES,
        },
        output_dir / "dynamic_policy_input_scaler.pkl",
    )

    # --------------------------------------------------------
    # 模仿预训练
    # --------------------------------------------------------
    imitation_history = imitation_pretrain(
        policy,
        train_loader,
        cfg,
        device,
    )

    # --------------------------------------------------------
    # 多步 DPC 训练
    # --------------------------------------------------------
    dpc_history = train_dpc(
        policy,
        dynamic_model,
        train_loader,
        val_loader,
        cfg,
        runtime,
        output_dir,
        input_mean,
        input_std,
        device,
    )

    # --------------------------------------------------------
    # 保存训练记录与配置
    # --------------------------------------------------------
    history = imitation_history + dpc_history
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    serializable_runtime = {
        "Ip_limit": float(runtime["Ip_limit"]),
        "Is_limit": float(runtime["Is_limit"]),
        "state_lower": runtime["state_lower"].tolist(),
        "state_upper": runtime["state_upper"].tolist(),
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(cfg, file, ensure_ascii=False, indent=2)

    with open(
        output_dir / "runtime_limits.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_runtime,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # 保存场景摘要
    scenario_summary = pd.DataFrame({
        "split": (
            ["train"] * len(train_arrays["x0"])
            + ["val"] * len(val_arrays["x0"])
        ),
        "scenario_type": np.concatenate([
            train_arrays["scenario_type"],
            val_arrays["scenario_type"],
        ]),
        "Vin_new": np.concatenate([
            train_arrays["condition"][:, 0],
            val_arrays["condition"][:, 0],
        ]),
        "Rload_new": np.concatenate([
            train_arrays["condition"][:, 1],
            val_arrays["condition"][:, 1],
        ]),
    })
    scenario_summary.to_csv(
        output_dir / "scenario_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 加载最佳策略并保存验证滚动结果
    # --------------------------------------------------------
    best_path = output_dir / "dynamic_policy_best.pth"
    if not best_path.exists():
        raise RuntimeError("未生成最佳策略模型。")

    best_checkpoint = load_torch_file(best_path)
    policy.load_state_dict(best_checkpoint["model_state_dict"])
    policy.eval()

    save_validation_rollouts(
        policy,
        dynamic_model,
        val_arrays,
        cfg,
        runtime,
        output_dir,
        device,
    )

    print("\n训练完成。")
    print(f"最佳模型：{best_path.resolve()}")
    print(
        "验证滚动："
        f"{(output_dir / 'validation_rollouts.csv').resolve()}"
    )
    print(
        "策略输入 scaler："
        f"{(output_dir / 'dynamic_policy_input_scaler.pkl').resolve()}"
    )
    print("\n下一步必须将该策略接回 PLECS 闭环验证。")


if __name__ == "__main__":
    main()
