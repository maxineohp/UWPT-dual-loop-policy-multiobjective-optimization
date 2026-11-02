# -*- coding: utf-8 -*-
"""
train_dynamic_surrogate_multistep.py

训练 WPT 动态代理模型，增加多步滚动训练功能。

输入 CSV 格式：
    time,Vin,Rload,a,b,c,Vout_avg,Pout_avg,eff_avg,Ip_rms,Is_rms

说明：
- eff_avg 不用于训练；
- Pout_avg 不用于训练；
- Pout 后续可由 Vout^2/Rload 计算；
- 代理模型仍然保持单步增量形式，便于和原 DPC 代码兼容；
- 训练阶段通过多步 rollout 损失提高模型长 horizon 滚动稳定性。

动态代理模型形式：
    input = [
        Vout(k), Ip_rms(k), Is_rms(k),
        a(k), b(k), c(k),
        da(k), db(k), dc(k),
        Vin(k+1), Rload(k+1)
    ]

    output = [
        dVout(k), dIp_rms(k), dIs_rms(k)
    ]

多步滚动训练：
    x_hat(k) = x(k)
    for i = 0 ... H-1:
        dx_hat(k+i) = model(x_hat(k+i), u(k+i), du(k+i), d(k+i+1))
        x_hat(k+i+1) = x_hat(k+i) + dx_hat(k+i)
        loss += ||x_hat(k+i+1) - x(k+i+1)||^2

输出文件夹：
    dynamic_surrogate_output_multistep/
        dynamic_surrogate.pth
        x_scaler.pkl
        y_scaler.pkl
        train_metrics.csv
        test_metrics.csv
        test_rollout_metrics.csv
        val_rollout_metrics.csv
        config.json
        split_info.json
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 配置
# ============================================================

CONFIG = {
    "data_dirs": [
        "traj_outputs",
    ],

    # 建议单独保存，避免覆盖原来的单步模型
    "output_dir": "dynamic_surrogate_output_multistep",

    "seed": 42,

    # 数据划分按“轨迹文件”划分，避免相邻样本泄漏
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # 网络结构。保持输入 11 维、输出 3 维，与原 DPC 代码兼容
    "hidden_dims": [256, 256, 128, 64],
    "dropout": 0.0,

    # 训练参数：先单步监督预训练，再多步 rollout 微调
    "batch_size": 1024,          # 单步监督预训练使用
    "enable_one_step_pretrain": True,
    "pretrain_epochs": 200,
    "pretrain_lr": 2e-5,
    "pretrain_patience": 50,

    "rollout_epochs": 600,
    "rollout_lr": 1e-5,
    "rollout_patience": 120,

    "weight_decay": 5e-7,
    "grad_clip": 5.0,

    # 多步滚动训练开关
    "enable_multistep_training": True,

    # H 的单位是控制步。若 Ts = 200 us，则：
    # 10 步 = 2 ms，15 步 = 3 ms，25 步 = 5 ms，40 步 = 8 ms
    # 按用户指定比例抽样：H=15 占 30%，H=25 占 30%，H=10 占 20%，H=40 占 20%。
    "multi_step_horizons": [10, 15, 25, 40],

    # 每个 batch 按权重随机抽一个 H。权重总和不要求为 1，程序内部会自动归一化。
    # 对应概率：H=10 -> 20%，H=15 -> 30%，H=25 -> 30%，H=40 -> 20%。
    "horizon_sampling_weights": [0.20, 0.30, 0.30, 0.20],

    # 多步 rollout batch 通常比单步 batch 小，否则显存/内存压力较大
    "rollout_batch_size": 256,

    # 构造 rollout 序列时的起点间隔。5 表示每隔 5 个采样点取一个起点，可显著加速训练
    "rollout_stride": 5,

    # 单个轨迹最多取多少条 rollout 序列。None 表示全取
    "max_rollout_sequences_per_file": None,

    # 多步损失归一化尺度。数值越小，该状态误差惩罚越重
    # 可根据你的系统量级调整：例如 Vout 5 V、Ip/Is 0.2 A 作为 1 个单位误差
    "rollout_loss_scales": {
        "Vout": 5.0,
        "Ip_rms": 0.2,
        "Is_rms": 0.2,
    },

    # 验证时评估这些 horizon 的 rollout loss
    "validation_horizons": [10, 15, 25, 40],

    # 测试时输出这些 horizon 的最终误差和全步平均误差
    "test_rollout_horizons": [10, 15, 25, 40],

    # rollout 验证/测试最多使用多少个 batch；None 表示全量评估
    "rollout_eval_max_batches": None,

    # 如果只想做原来的单步训练，把 enable_multistep_training 改成 False
    "single_step_loss_weight": 1.0,

    # 是否删除明显异常数据
    "enable_basic_filter": True,

    # 物理范围过滤，只是防止仿真炸掉的数据进入训练
    "filter": {
        "Vout_min": -1e6,
        "Vout_max": 1e6,
        "Ip_max": 1e6,
        "Is_max": 1e6,
        "abs_dVout_max": 1e6,
        "abs_dIp_max": 1e6,
        "abs_dIs_max": 1e6,
    },

    # 时间步检查。若你的 DPC 控制周期是 200 us，这里建议保持 200e-6
    # 只报警，不删除数据
    "enable_dt_warning": True,
    "expected_dt": 200e-6,
    "dt_warn_tol": 1e-9,
}


# ============================================================
# 2. 随机种子
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. 模型
# ============================================================

class DynamicSurrogateMLP(nn.Module):
    def __init__(self, input_dim: int = 11, output_dim: int = 3,
                 hidden_dims: Optional[Sequence[int]] = None, dropout: float = 0.0):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        layers: List[nn.Module] = []
        last_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, int(h)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            last_dim = int(h)

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Dataset
# ============================================================

class ArrayDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class RolloutDataset(Dataset):
    """
    返回连续片段，形状为 [max_horizon + 1, 8]。
    8 列依次为：
        Vout_avg, Ip_rms, Is_rms, a, b, c, Vin, Rload
    """
    def __init__(
        self,
        arrays: Sequence[np.ndarray],
        max_horizon: int,
        stride: int = 1,
        max_sequences_per_file: Optional[int] = None,
        seed: int = 42,
    ):
        self.max_horizon = int(max_horizon)
        self.arrays = [torch.as_tensor(a, dtype=torch.float32) for a in arrays if len(a) >= self.max_horizon + 1]
        self.index: List[Tuple[int, int]] = []

        rng = np.random.default_rng(seed)
        stride = max(int(stride), 1)

        for traj_idx, arr in enumerate(self.arrays):
            n = arr.shape[0]
            starts = np.arange(0, n - self.max_horizon, stride, dtype=np.int64)
            if max_sequences_per_file is not None and len(starts) > int(max_sequences_per_file):
                starts = rng.choice(starts, size=int(max_sequences_per_file), replace=False)
                starts = np.sort(starts)
            for s in starts.tolist():
                self.index.append((traj_idx, int(s)))

        if len(self.index) == 0:
            raise RuntimeError("没有可用于多步 rollout 的序列。请检查轨迹长度或 multi_step_horizons。")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        traj_idx, start = self.index[idx]
        return self.arrays[traj_idx][start:start + self.max_horizon + 1]


# ============================================================
# 5. 数据读取与样本构造
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

ROLLOUT_COLUMNS = [
    "Vout_avg",
    "Ip_rms",
    "Is_rms",
    "a",
    "b",
    "c",
    "Vin",
    "Rload",
]

INPUT_NAMES = [
    "Vout",
    "Ip_rms",
    "Is_rms",
    "a",
    "b",
    "c",
    "da",
    "db",
    "dc",
    "Vin_next",
    "Rload_next",
]

OUTPUT_NAMES = [
    "dVout",
    "dIp_rms",
    "dIs_rms",
]

STATE_NAMES = ["Vout", "Ip_rms", "Is_rms"]


def find_csv_files(data_dirs):
    base = Path.cwd()
    files = []

    for d in data_dirs:
        p = base / d
        if not p.exists():
            print(f"警告：数据目录不存在，跳过：{p}")
            continue

        files.extend(sorted(p.glob("*.csv")))

    files = sorted(files, key=lambda x: str(x))
    return files


def read_one_csv(path: Path, cfg=None):
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列：{missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.sort_values("time").reset_index(drop=True)

    # 删除重复时间点
    df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    # 全部转成数值
    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if cfg is not None and bool(cfg.get("enable_dt_warning", False)) and len(df) >= 3:
        dt = df["time"].diff().dropna().to_numpy(dtype=np.float64)
        expected_dt = float(cfg.get("expected_dt", 0.0))
        tol = float(cfg.get("dt_warn_tol", 1e-9))
        if expected_dt > 0 and (np.nanmin(np.abs(dt - expected_dt)) > tol or np.nanmax(np.abs(dt - expected_dt)) > tol):
            print(
                f"警告：{path.name} 时间步可能不是 {expected_dt:.6g} s，"
                f"dt_min={np.nanmin(dt):.6g}, dt_max={np.nanmax(dt):.6g}, dt_mean={np.nanmean(dt):.6g}"
            )

    return df


def build_samples_from_df(df: pd.DataFrame, cfg):
    """
    用相邻两行构造：
        input(k)  = [Vout, Ip, Is, a,b,c, da,db,dc, Vin_next, R_next]
        target(k) = [dVout, dIp, dIs]
    """
    if len(df) < 2:
        return None, None

    cur = df.iloc[:-1].reset_index(drop=True)
    nxt = df.iloc[1:].reset_index(drop=True)

    Vout_k = cur["Vout_avg"].to_numpy(dtype=np.float64)
    Ip_k = cur["Ip_rms"].to_numpy(dtype=np.float64)
    Is_k = cur["Is_rms"].to_numpy(dtype=np.float64)

    a_k = cur["a"].to_numpy(dtype=np.float64)
    b_k = cur["b"].to_numpy(dtype=np.float64)
    c_k = cur["c"].to_numpy(dtype=np.float64)

    a_next = nxt["a"].to_numpy(dtype=np.float64)
    b_next = nxt["b"].to_numpy(dtype=np.float64)
    c_next = nxt["c"].to_numpy(dtype=np.float64)

    da = a_next - a_k
    db = b_next - b_k
    dc = c_next - c_k

    # 采用第 k+1 行的 Vin/Rload 作为 k→k+1 区间实际施加的工况
    Vin_next = nxt["Vin"].to_numpy(dtype=np.float64)
    R_next = nxt["Rload"].to_numpy(dtype=np.float64)

    Vout_next = nxt["Vout_avg"].to_numpy(dtype=np.float64)
    Ip_next = nxt["Ip_rms"].to_numpy(dtype=np.float64)
    Is_next = nxt["Is_rms"].to_numpy(dtype=np.float64)

    dVout = Vout_next - Vout_k
    dIp = Ip_next - Ip_k
    dIs = Is_next - Is_k

    X = np.column_stack([
        Vout_k,
        Ip_k,
        Is_k,
        a_k,
        b_k,
        c_k,
        da,
        db,
        dc,
        Vin_next,
        R_next,
    ])

    Y = np.column_stack([
        dVout,
        dIp,
        dIs,
    ])

    if cfg["enable_basic_filter"]:
        f = cfg["filter"]

        mask = np.ones(len(X), dtype=bool)
        mask &= np.isfinite(X).all(axis=1)
        mask &= np.isfinite(Y).all(axis=1)
        mask &= Vout_k >= f["Vout_min"]
        mask &= Vout_k <= f["Vout_max"]
        mask &= np.abs(Ip_k) <= f["Ip_max"]
        mask &= np.abs(Is_k) <= f["Is_max"]
        mask &= np.abs(dVout) <= f["abs_dVout_max"]
        mask &= np.abs(dIp) <= f["abs_dIp_max"]
        mask &= np.abs(dIs) <= f["abs_dIs_max"]

        X = X[mask]
        Y = Y[mask]

    return X.astype(np.float32), Y.astype(np.float32)


def build_rollout_array_from_df(df: pd.DataFrame, cfg):
    if len(df) < 2:
        return None

    arr = df[ROLLOUT_COLUMNS].to_numpy(dtype=np.float32)

    if cfg["enable_basic_filter"]:
        f = cfg["filter"]
        mask = np.ones(len(arr), dtype=bool)
        mask &= np.isfinite(arr).all(axis=1)
        mask &= arr[:, 0] >= f["Vout_min"]
        mask &= arr[:, 0] <= f["Vout_max"]
        mask &= np.abs(arr[:, 1]) <= f["Ip_max"]
        mask &= np.abs(arr[:, 2]) <= f["Is_max"]
        arr = arr[mask]

    if len(arr) < 2:
        return None

    return arr.astype(np.float32)


def split_files_by_trajectory(files, cfg):
    files = list(files)
    random.shuffle(files)

    n = len(files)
    n_train = int(round(n * cfg["train_ratio"]))
    n_val = int(round(n * cfg["val_ratio"]))

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def load_dataset_from_files(files, cfg, name="dataset"):
    X_list = []
    Y_list = []
    used_files = []
    skipped = []

    for path in files:
        try:
            df = read_one_csv(path, cfg)
            X, Y = build_samples_from_df(df, cfg)

            if X is None or len(X) == 0:
                skipped.append((str(path), "no_valid_samples"))
                continue

            X_list.append(X)
            Y_list.append(Y)
            used_files.append(str(path))

        except Exception as e:
            skipped.append((str(path), str(e)))

    if len(X_list) == 0:
        raise RuntimeError(f"{name} 没有有效样本。")

    X_all = np.vstack(X_list)
    Y_all = np.vstack(Y_list)

    print(f"{name}:")
    print(f"  files used    = {len(used_files)}")
    print(f"  files skipped = {len(skipped)}")
    print(f"  samples       = {len(X_all)}")

    if skipped:
        print(f"  前 5 个跳过文件：")
        for item in skipped[:5]:
            print(f"    {item[0]} | {item[1]}")

    return X_all, Y_all, used_files, skipped


def load_rollout_arrays_from_files(files, cfg, name="rollout"):
    arrays = []
    used_files = []
    skipped = []

    max_horizon = int(max(cfg["multi_step_horizons"]))

    for path in files:
        try:
            df = read_one_csv(path, cfg)
            arr = build_rollout_array_from_df(df, cfg)

            if arr is None or len(arr) < max_horizon + 1:
                skipped.append((str(path), f"length < max_horizon + 1 = {max_horizon + 1}"))
                continue

            arrays.append(arr)
            used_files.append(str(path))

        except Exception as e:
            skipped.append((str(path), str(e)))

    if len(arrays) == 0:
        raise RuntimeError(f"{name} 没有有效 rollout 轨迹。")

    total_points = sum(len(a) for a in arrays)
    print(f"{name} rollout:")
    print(f"  files used    = {len(used_files)}")
    print(f"  files skipped = {len(skipped)}")
    print(f"  points        = {total_points}")

    if skipped:
        print(f"  前 5 个跳过文件：")
        for item in skipped[:5]:
            print(f"    {item[0]} | {item[1]}")

    return arrays, used_files, skipped


# ============================================================
# 6. 训练与评估
# ============================================================

def compute_metrics_np(y_true, y_pred, names):
    err = y_pred - y_true

    rows = []
    for i, name in enumerate(names):
        mae = np.mean(np.abs(err[:, i]))
        rmse = math.sqrt(np.mean(err[:, i] ** 2))
        maxe = np.max(np.abs(err[:, i]))

        denom = np.maximum(np.abs(y_true[:, i]), 1e-9)
        mape = np.mean(np.abs(err[:, i]) / denom) * 100.0

        rows.append({
            "target": name,
            "MAE": mae,
            "RMSE": rmse,
            "MaxAbs": maxe,
            "MAPE_percent": mape,
        })

    return pd.DataFrame(rows)


def evaluate_one_step(model, loader, device):
    model.eval()

    y_true_all = []
    y_pred_all = []

    total_loss = 0.0
    n_total = 0

    loss_fn = nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            total_loss += loss.item()
            n_total += yb.numel()

            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

    mse = total_loss / max(n_total, 1)

    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)

    return mse, y_true, y_pred


def make_scaler_tensors(x_scaler, y_scaler, device):
    x_mean = torch.as_tensor(x_scaler.mean_, dtype=torch.float32, device=device)
    x_scale = torch.as_tensor(x_scaler.scale_, dtype=torch.float32, device=device).clamp_min(1e-12)
    y_mean = torch.as_tensor(y_scaler.mean_, dtype=torch.float32, device=device)
    y_scale = torch.as_tensor(y_scaler.scale_, dtype=torch.float32, device=device).clamp_min(1e-12)
    return x_mean, x_scale, y_mean, y_scale


def get_state_loss_scale_tensor(cfg, device):
    s = cfg["rollout_loss_scales"]
    return torch.as_tensor(
        [float(s["Vout"]), float(s["Ip_rms"]), float(s["Is_rms"])],
        dtype=torch.float32,
        device=device,
    ).clamp_min(1e-12)


def sample_horizon(cfg):
    horizons = np.asarray(cfg["multi_step_horizons"], dtype=np.int64)
    weights = np.asarray(cfg.get("horizon_sampling_weights", np.ones_like(horizons)), dtype=np.float64)
    if len(weights) != len(horizons):
        raise ValueError("horizon_sampling_weights 的长度必须与 multi_step_horizons 一致。")
    weights = weights / np.sum(weights)
    return int(np.random.choice(horizons, p=weights))


def rollout_loss_batch(model, seq, horizon, x_mean, x_scale, y_mean, y_scale, state_loss_scale):
    """
    seq: [B, max_horizon + 1, 8]
         8 列 = Vout, Ip, Is, a, b, c, Vin, Rload

    返回：平均多步状态预测损失。
    """
    H = int(horizon)
    state_pred = seq[:, 0, 0:3]  # [B,3]

    total_loss = 0.0

    for t in range(H):
        angle_now = seq[:, t, 3:6]
        angle_next = seq[:, t + 1, 3:6]
        d_angle = angle_next - angle_now
        vin_r_next = seq[:, t + 1, 6:8]

        x_phys = torch.cat([state_pred, angle_now, d_angle, vin_r_next], dim=1)
        x_scaled = (x_phys - x_mean) / x_scale

        dy_scaled = model(x_scaled)
        dy_phys = dy_scaled * y_scale + y_mean

        state_pred = state_pred + dy_phys
        state_true_next = seq[:, t + 1, 0:3]

        err_norm = (state_pred - state_true_next) / state_loss_scale
        total_loss = total_loss + torch.mean(err_norm ** 2)

    return total_loss / max(H, 1)


def evaluate_rollout_loss_multi(model, loader, device, cfg, x_scaler, y_scaler):
    model.eval()
    horizons = [int(h) for h in cfg["validation_horizons"]]
    max_batches = cfg.get("rollout_eval_max_batches", None)

    x_mean, x_scale, y_mean, y_scale = make_scaler_tensors(x_scaler, y_scaler, device)
    state_loss_scale = get_state_loss_scale_tensor(cfg, device)

    total = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, seq in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            seq = seq.to(device)
            batch_loss = 0.0
            for H in horizons:
                batch_loss = batch_loss + rollout_loss_batch(
                    model, seq, H, x_mean, x_scale, y_mean, y_scale, state_loss_scale
                )
            batch_loss = batch_loss / max(len(horizons), 1)
            total += float(batch_loss.item()) * seq.shape[0]
            count += seq.shape[0]

    return total / max(count, 1)


def evaluate_rollout_metrics(model, loader, device, cfg, x_scaler, y_scaler, horizons):
    """
    输出每个 horizon 下：
    - final_MAE/RMSE：第 H 步最终状态误差
    - mean_step_MAE/RMSE：1...H 全部滚动步的平均状态误差
    """
    model.eval()
    max_batches = cfg.get("rollout_eval_max_batches", None)

    x_mean, x_scale, y_mean, y_scale = make_scaler_tensors(x_scaler, y_scaler, device)

    rows = []

    with torch.no_grad():
        for H in [int(h) for h in horizons]:
            final_abs_sum = torch.zeros(3, dtype=torch.float64, device=device)
            final_sq_sum = torch.zeros(3, dtype=torch.float64, device=device)
            step_abs_sum = torch.zeros(3, dtype=torch.float64, device=device)
            step_sq_sum = torch.zeros(3, dtype=torch.float64, device=device)
            final_n = 0
            step_n = 0

            for batch_idx, seq in enumerate(loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                seq = seq.to(device)
                state_pred = seq[:, 0, 0:3]

                for t in range(H):
                    angle_now = seq[:, t, 3:6]
                    angle_next = seq[:, t + 1, 3:6]
                    d_angle = angle_next - angle_now
                    vin_r_next = seq[:, t + 1, 6:8]

                    x_phys = torch.cat([state_pred, angle_now, d_angle, vin_r_next], dim=1)
                    x_scaled = (x_phys - x_mean) / x_scale
                    dy_scaled = model(x_scaled)
                    dy_phys = dy_scaled * y_scale + y_mean
                    state_pred = state_pred + dy_phys

                    state_true_next = seq[:, t + 1, 0:3]
                    err = (state_pred - state_true_next).to(torch.float64)
                    step_abs_sum += torch.sum(torch.abs(err), dim=0)
                    step_sq_sum += torch.sum(err ** 2, dim=0)
                    step_n += err.shape[0]

                final_err = (state_pred - seq[:, H, 0:3]).to(torch.float64)
                final_abs_sum += torch.sum(torch.abs(final_err), dim=0)
                final_sq_sum += torch.sum(final_err ** 2, dim=0)
                final_n += final_err.shape[0]

            for i, name in enumerate(STATE_NAMES):
                rows.append({
                    "horizon": H,
                    "target": name,
                    "final_MAE": float(final_abs_sum[i].item() / max(final_n, 1)),
                    "final_RMSE": float(math.sqrt(final_sq_sum[i].item() / max(final_n, 1))),
                    "mean_step_MAE": float(step_abs_sum[i].item() / max(step_n, 1)),
                    "mean_step_RMSE": float(math.sqrt(step_sq_sum[i].item() / max(step_n, 1))),
                })

    return pd.DataFrame(rows)


def train_one_step_epoch(model, loader, optimizer, device, cfg):
    model.train()
    loss_fn = nn.MSELoss()
    train_loss_sum = 0.0
    train_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()

        if cfg["grad_clip"] is not None and cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))

        optimizer.step()

        train_loss_sum += loss.item() * xb.shape[0]
        train_count += xb.shape[0]

    return train_loss_sum / max(train_count, 1)


def train_multistep_epoch(model, loader, optimizer, device, cfg, x_scaler, y_scaler):
    model.train()

    x_mean, x_scale, y_mean, y_scale = make_scaler_tensors(x_scaler, y_scaler, device)
    state_loss_scale = get_state_loss_scale_tensor(cfg, device)

    train_loss_sum = 0.0
    train_count = 0
    horizon_counter: Dict[int, int] = {}

    for seq in loader:
        seq = seq.to(device)
        H = sample_horizon(cfg)
        horizon_counter[H] = horizon_counter.get(H, 0) + 1

        optimizer.zero_grad()
        loss = rollout_loss_batch(model, seq, H, x_mean, x_scale, y_mean, y_scale, state_loss_scale)
        loss.backward()

        if cfg["grad_clip"] is not None and cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))

        optimizer.step()

        train_loss_sum += loss.item() * seq.shape[0]
        train_count += seq.shape[0]

    return train_loss_sum / max(train_count, 1), horizon_counter


def train_model(cfg):
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    files = find_csv_files(cfg["data_dirs"])
    if not files:
        raise RuntimeError("没有找到任何 CSV 文件，请检查 data_dirs。")

    print(f"Total CSV files found: {len(files)}")

    train_files, val_files, test_files = split_files_by_trajectory(files, cfg)

    # 单步样本用于 scaler 拟合、单步评价，以及 enable_multistep_training=False 时的训练
    X_train, Y_train, used_train, skipped_train = load_dataset_from_files(train_files, cfg, "train")
    X_val, Y_val, used_val, skipped_val = load_dataset_from_files(val_files, cfg, "val")
    X_test, Y_test, used_test, skipped_test = load_dataset_from_files(test_files, cfg, "test")

    # 标准化器仍然按单步输入/输出拟合，保持原模型接口
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    Y_train_s = y_scaler.fit_transform(Y_train).astype(np.float32)
    X_val_s = x_scaler.transform(X_val).astype(np.float32)
    Y_val_s = y_scaler.transform(Y_val).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)
    Y_test_s = y_scaler.transform(Y_test).astype(np.float32)

    joblib.dump(x_scaler, output_dir / "x_scaler.pkl")
    joblib.dump(y_scaler, output_dir / "y_scaler.pkl")

    train_loader = DataLoader(
        ArrayDataset(X_train_s, Y_train_s),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        ArrayDataset(X_val_s, Y_val_s),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        ArrayDataset(X_test_s, Y_test_s),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
    )

    train_rollout_loader = None
    val_rollout_loader = None
    test_rollout_loader = None
    used_train_rollout = []
    used_val_rollout = []
    used_test_rollout = []
    skipped_train_rollout = []
    skipped_val_rollout = []
    skipped_test_rollout = []

    if bool(cfg.get("enable_multistep_training", False)):
        max_horizon = int(max(cfg["multi_step_horizons"]))
        print(f"Multi-step training enabled. max_horizon = {max_horizon}")

        train_arrays, used_train_rollout, skipped_train_rollout = load_rollout_arrays_from_files(train_files, cfg, "train")
        val_arrays, used_val_rollout, skipped_val_rollout = load_rollout_arrays_from_files(val_files, cfg, "val")
        test_arrays, used_test_rollout, skipped_test_rollout = load_rollout_arrays_from_files(test_files, cfg, "test")

        train_rollout_ds = RolloutDataset(
            train_arrays,
            max_horizon=max_horizon,
            stride=int(cfg["rollout_stride"]),
            max_sequences_per_file=cfg.get("max_rollout_sequences_per_file", None),
            seed=int(cfg["seed"]),
        )
        val_rollout_ds = RolloutDataset(
            val_arrays,
            max_horizon=max_horizon,
            stride=int(cfg["rollout_stride"]),
            max_sequences_per_file=cfg.get("max_rollout_sequences_per_file", None),
            seed=int(cfg["seed"]) + 1,
        )
        test_rollout_ds = RolloutDataset(
            test_arrays,
            max_horizon=max_horizon,
            stride=int(cfg["rollout_stride"]),
            max_sequences_per_file=cfg.get("max_rollout_sequences_per_file", None),
            seed=int(cfg["seed"]) + 2,
        )

        print(f"rollout train sequences = {len(train_rollout_ds)}")
        print(f"rollout val sequences   = {len(val_rollout_ds)}")
        print(f"rollout test sequences  = {len(test_rollout_ds)}")

        train_rollout_loader = DataLoader(
            train_rollout_ds,
            batch_size=int(cfg["rollout_batch_size"]),
            shuffle=True,
            drop_last=False,
        )
        val_rollout_loader = DataLoader(
            val_rollout_ds,
            batch_size=int(cfg["rollout_batch_size"]),
            shuffle=False,
            drop_last=False,
        )
        test_rollout_loader = DataLoader(
            test_rollout_ds,
            batch_size=int(cfg["rollout_batch_size"]),
            shuffle=False,
            drop_last=False,
        )

    model = DynamicSurrogateMLP(
        input_dim=11,
        output_dim=3,
        hidden_dims=cfg["hidden_dims"],
        dropout=float(cfg["dropout"]),
    ).to(device)

    history = []
    best_path = output_dir / "dynamic_surrogate.pth"
    pretrain_path = output_dir / "dynamic_surrogate_pretrain.pth"

    def save_checkpoint(path, stage, epoch, val_loss):
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "input_dim": 11,
            "output_dim": 3,
            "input_names": INPUT_NAMES,
            "output_names": OUTPUT_NAMES,
            "hidden_dims": cfg["hidden_dims"],
            "training_mode": stage,
            "best_epoch": int(epoch),
            "best_val_loss": float(val_loss),
        }, path)

    # ------------------------------------------------------------
    # Stage 1: 单步监督预训练
    # ------------------------------------------------------------
    best_val = float("inf")
    best_epoch = -1
    best_stage = "none"

    if bool(cfg.get("enable_one_step_pretrain", True)):
        print("\n========== Stage 1: one-step supervised pretraining ==========")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["pretrain_lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )

        pretrain_best_val = float("inf")
        pretrain_best_epoch = -1
        wait = 0

        for epoch in range(1, int(cfg["pretrain_epochs"]) + 1):
            train_loss = train_one_step_epoch(model, train_loader, optimizer, device, cfg)
            val_loss, _, _ = evaluate_one_step(model, val_loader, device)

            history.append({
                "stage": "pretrain",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "horizon_info": "single-step supervised",
            })

            if epoch == 1 or epoch % 10 == 0:
                print(f"[Pretrain] Epoch {epoch:04d} | train={train_loss:.6e} | val={val_loss:.6e}")

            if val_loss < pretrain_best_val:
                pretrain_best_val = val_loss
                pretrain_best_epoch = epoch
                wait = 0
                save_checkpoint(pretrain_path, "one_step_pretrain", epoch, val_loss)
                # 如果后续关闭多步训练，则该文件就是最终模型
                save_checkpoint(best_path, "one_step_pretrain", epoch, val_loss)
            else:
                wait += 1

            if wait >= int(cfg["pretrain_patience"]):
                print(
                    f"[Pretrain] Early stopping at epoch {epoch}. "
                    f"Best epoch = {pretrain_best_epoch}, val = {pretrain_best_val:.6e}"
                )
                break

        if pretrain_path.exists():
            ckpt = torch.load(pretrain_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(
                f"[Pretrain] Best epoch = {pretrain_best_epoch}, "
                f"best val = {pretrain_best_val:.6e}. Loaded for rollout fine-tuning."
            )
            best_val = pretrain_best_val
            best_epoch = pretrain_best_epoch
            best_stage = "one_step_pretrain"
    else:
        print("\nSkip one-step supervised pretraining.")

    # ------------------------------------------------------------
    # Stage 2: 多步 rollout 微调
    # ------------------------------------------------------------
    if bool(cfg.get("enable_multistep_training", False)):
        print("\n========== Stage 2: multi-step rollout fine-tuning ==========")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["rollout_lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )

        rollout_best_val = float("inf")
        rollout_best_epoch = -1
        wait = 0

        for epoch in range(1, int(cfg["rollout_epochs"]) + 1):
            train_loss, horizon_counter = train_multistep_epoch(
                model, train_rollout_loader, optimizer, device, cfg, x_scaler, y_scaler
            )
            val_loss = evaluate_rollout_loss_multi(
                model, val_rollout_loader, device, cfg, x_scaler, y_scaler
            )
            horizon_info = ", ".join([
                f"H{h}:{horizon_counter.get(int(h), 0)}" for h in cfg["multi_step_horizons"]
            ])

            history.append({
                "stage": "rollout",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "horizon_info": horizon_info,
            })

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"[Rollout] Epoch {epoch:04d} | train={train_loss:.6e} | "
                    f"val={val_loss:.6e} | {horizon_info}"
                )

            if val_loss < rollout_best_val:
                rollout_best_val = val_loss
                rollout_best_epoch = epoch
                wait = 0
                save_checkpoint(best_path, "multistep_rollout", epoch, val_loss)
            else:
                wait += 1

            if wait >= int(cfg["rollout_patience"]):
                print(
                    f"[Rollout] Early stopping at epoch {epoch}. "
                    f"Best epoch = {rollout_best_epoch}, val = {rollout_best_val:.6e}"
                )
                break

        best_val = rollout_best_val
        best_epoch = rollout_best_epoch
        best_stage = "multistep_rollout"
    else:
        print("\nSkip multi-step rollout fine-tuning.")

    # 保存训练曲线
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "train_metrics.csv", index=False)

    # 加载最佳模型
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 单步测试集评估，注意要反标准化
    test_mse_scaled, y_test_s_true, y_test_s_pred = evaluate_one_step(model, test_loader, device)
    y_test_true = y_scaler.inverse_transform(y_test_s_true)
    y_test_pred = y_scaler.inverse_transform(y_test_s_pred)

    metrics_df = compute_metrics_np(
        y_true=y_test_true,
        y_pred=y_test_pred,
        names=OUTPUT_NAMES,
    )
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)

    pred_df = pd.DataFrame({
        "dVout_true": y_test_true[:, 0],
        "dVout_pred": y_test_pred[:, 0],
        "dIp_true": y_test_true[:, 1],
        "dIp_pred": y_test_pred[:, 1],
        "dIs_true": y_test_true[:, 2],
        "dIs_pred": y_test_pred[:, 2],
    })
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # 多步 rollout 评估
    if bool(cfg.get("enable_multistep_training", False)):
        val_rollout_metrics = evaluate_rollout_metrics(
            model, val_rollout_loader, device, cfg, x_scaler, y_scaler, cfg["test_rollout_horizons"]
        )
        test_rollout_metrics = evaluate_rollout_metrics(
            model, test_rollout_loader, device, cfg, x_scaler, y_scaler, cfg["test_rollout_horizons"]
        )
        val_rollout_metrics.to_csv(output_dir / "val_rollout_metrics.csv", index=False)
        test_rollout_metrics.to_csv(output_dir / "test_rollout_metrics.csv", index=False)
    else:
        val_rollout_metrics = None
        test_rollout_metrics = None

    # 保存划分信息
    split_info = {
        "train_files": used_train,
        "val_files": used_val,
        "test_files": used_test,
        "skipped_train": skipped_train,
        "skipped_val": skipped_val,
        "skipped_test": skipped_test,
        "train_rollout_files": used_train_rollout,
        "val_rollout_files": used_val_rollout,
        "test_rollout_files": used_test_rollout,
        "skipped_train_rollout": skipped_train_rollout,
        "skipped_val_rollout": skipped_val_rollout,
        "skipped_test_rollout": skipped_test_rollout,
    }

    with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\n训练完成。")
    print(f"Best stage: {best_stage}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val:.6e}")
    print(f"Test one-step MSE scaled: {test_mse_scaled:.6e}")
    print(f"模型保存到: {best_path.resolve()}")
    print(f"单步测试指标保存到: {(output_dir / 'test_metrics.csv').resolve()}")

    if bool(cfg.get("enable_multistep_training", False)):
        print(f"多步验证指标保存到: {(output_dir / 'val_rollout_metrics.csv').resolve()}")
        print(f"多步测试指标保存到: {(output_dir / 'test_rollout_metrics.csv').resolve()}")

    print("\n单步测试集指标：")
    print(metrics_df)

    if test_rollout_metrics is not None:
        print("\n多步 rollout 测试集指标：")
        print(test_rollout_metrics)


# ============================================================
# 7. 主入口
# ============================================================

if __name__ == "__main__":
    train_model(CONFIG)
