# -*- coding: utf-8 -*-
"""批量生成 NSGA-II 偏好标签；不训练稳态策略网络。"""

import os
import json
import copy
import math
import time
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CONFIG = {
    # ---------- 基础文件 ----------
    "csv_path": "data/merged_all.csv",
    "model_dir": "model_output_dual_no_pout",
    "model_eff_path": os.path.join("model_output_dual_no_pout", "model_eff.pth"),
    "model_aux_path": os.path.join("model_output_dual_no_pout", "model_aux.pth"),
    "x_scaler_path": os.path.join("model_output_dual_no_pout", "x_scaler_shared.pkl"),
    "y_scaler_eff_path": os.path.join("model_output_dual_no_pout", "y_scaler_eff.pkl"),
    "y_scaler_aux_path": os.path.join("model_output_dual_no_pout", "y_scaler_aux.pkl"),

    # ---------- 当前 merged_all.csv 列定义 ----------
    # 当前表头：
    # case_id,status,error_msg,V,R,a,b,c,Vout_avg_last,
    # Pout_avg_last,eff_avg_last,Ip_rms_last,Is_rms_last
    "column_candidates": {
        "Vin": ["V", "Vin"],
        "Rload": ["R", "Rload"],
        "a": ["a"],
        "b": ["b"],
        "c": ["c"],
    },
    "status_column": "status",
    "valid_status": "OK",
    "input_names": ["Vin", "Rload", "a", "b", "c"],

    # ---------- 代理模型结构：必须和 train_surrogate_no_pout.py 一致 ----------
    "use_float64": True,
    "use_cuda_if_available": True,
    "eff_hidden_dims": [256, 256, 128, 64],
    "eff_dropout": 0.0,
    "aux_hidden_dims": [256, 256, 128, 64],
    "aux_dropout": 0.0,

    # ---------- 稳态控制目标 ----------
    # 当前优化输出电压要求：200 V ± 5 V，即 195~205 V
    "Vout_ref": 200.0,
    "Vout_tol_nsga": 3.0,
    "Vout_tol_finetune": 1.0,

    # ---------- NSGA-II 标签生成 ----------
    "random_seed": 42,
    "n_operating_points": 120,
    "op_sampling": "uniform_in_csv_range",  # Vin 连续采样，Rload 按下方指定范围采样

    # 当前优化负载范围：100 Ω ± 40 Ω，即 60~140 Ω
    "rload_min": 80.0,
    "rload_max": 130.0,

    "preference_values": [0.0, 0.25, 0.5, 0.75, 1.0],
    # 直接读取已生成的 NSGA 监督标签，不重新运行 NSGA
    "nsga_label_path": os.path.join("steady_policy_nsga_pref_no_pout_output", "nsga_pref_labels.csv"),

    "nsga_population_size": 160,
    "nsga_num_generations": 100,
    "nsga_crossover_prob": 0.9,
    "nsga_mutation_prob": 0.25,
    "nsga_mutation_sigma_ratio": 0.10,
    "nsga_archive_max_size": 120,
    "nsga_print_every_case": 1,

    # ---------- 并行设置 ----------
    # True: 按 Vin/Rload 工况多进程并行生成 NSGA 标签。
    # False: 单进程顺序生成，便于调试。
    "parallel_enabled": True,

    # worker 数量。可填整数，也可填 "auto"。
    # "auto" 会按 CPU 核心数自动选择，但不超过 max_parallel_workers。
    "n_workers": "auto",
    "max_parallel_workers": 30,

    # 每个 worker 内部 torch / BLAS 线程数。多进程时建议保持 1，避免线程过度竞争。
    "torch_num_threads_per_worker": 1,

    # 多进程并行时建议强制 CPU，避免多个进程同时抢同一张 GPU。
    "force_cpu_in_parallel": True,

    # a,b,c 是否按固定步长量化；连续优化保持 None
    "a_step": None,
    "b_step": None,
    "c_step": None,

    # ---------- 策略网络结构 ----------
    "policy_hidden_dims": [128, 128, 64, 32],
    "policy_dropout": 0.0,
    "policy_activation": "relu",

    # ---------- 重点工况加权 ----------
    # 让 Vin=300±50 V、Rload=100~150 Ω 区域的样本在监督预训练和自训练中权重更高。
    # 这不会重新运行 NSGA，也不会生成新标签；只改变 loss 中各样本的权重。
    "focus_region_enabled": True,
    "focus_vin_min": 150.0,
    "focus_vin_max": 250.0,
    "focus_rload_min": 60.0,
    "focus_rload_max": 140.0,
    "focus_sample_weight": 10.0,
    "normal_sample_weight": 1.0,

    # ---------- 监督预训练 ----------
    "val_ratio": 0.15,
    "batch_size": 128,
    "pretrain_epochs": 600,
    "pretrain_lr": 1e-3,
    "pretrain_weight_decay": 1e-6,
    "pretrain_patience": 80,

    # ---------- 代理模型自训练微调 ----------
    "finetune_epochs": 800,
    "finetune_lr": 3e-4,
    "finetune_weight_decay": 1e-7,
    "finetune_patience": 120,

    # 微调损失权重：pref_eff 越大，效率权重越大；pref_eff 越小，Irms 权重越大。
    "ft_weight_eff_min": 0.25,
    "ft_weight_eff_max": 1.00,
    "ft_weight_vout": 80.0,
    "ft_weight_irms_min": 0.02,
    "ft_weight_irms_max": 0.25,
    "ft_weight_angle_anchor_start": 1.0,
    "ft_weight_angle_anchor_end": 0.05,

    # efficiency 如果是 0~1，则自动用 1；如果是 0~100，则自动用 100。
    "eff_divisor": "auto",
    # Irms 归一化基准，None 表示用 NSGA 标签中的 Irms_obj 中位数。
    "irms_ref": None,

    # ---------- 输出 ----------
    "save_dir": "steady_policy_nsga_pref_no_pout_output",
    "save_plots": True,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, map_location="cpu", weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_dtype_and_device(cfg):
    if cfg["use_float64"]:
        torch.set_default_dtype(torch.float64)
        np_dtype = np.float64
        torch_dtype = torch.float64
    else:
        torch.set_default_dtype(torch.float32)
        np_dtype = np.float32
        torch_dtype = torch.float32

    if cfg["use_cuda_if_available"] and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return np_dtype, torch_dtype, device


def ensure_files_exist(paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256, 128, 64), dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchScaler:
    def __init__(self, sklearn_scaler, device, dtype):
        if hasattr(sklearn_scaler, "mean_") and hasattr(sklearn_scaler, "scale_"):
            self.mode = "standard"
            self.mean = torch.tensor(sklearn_scaler.mean_, device=device, dtype=dtype)
            self.scale = torch.tensor(sklearn_scaler.scale_, device=device, dtype=dtype)
            self.scale = torch.where(torch.abs(self.scale) < 1e-12, torch.ones_like(self.scale), self.scale)
        elif hasattr(sklearn_scaler, "min_") and hasattr(sklearn_scaler, "scale_"):
            self.mode = "minmax"
            self.min = torch.tensor(sklearn_scaler.min_, device=device, dtype=dtype)
            self.scale = torch.tensor(sklearn_scaler.scale_, device=device, dtype=dtype)
            self.scale = torch.where(torch.abs(self.scale) < 1e-12, torch.ones_like(self.scale), self.scale)
        else:
            raise TypeError("当前 scaler 不是 StandardScaler 或 MinMaxScaler，无法进行可微自训练。")

    def transform(self, x):
        if self.mode == "standard":
            return (x - self.mean) / self.scale
        if self.mode == "minmax":
            return x * self.scale + self.min
        raise RuntimeError("Unknown scaler mode")

    def inverse_transform(self, x):
        if self.mode == "standard":
            return x * self.scale + self.mean
        if self.mode == "minmax":
            return (x - self.min) / self.scale
        raise RuntimeError("Unknown scaler mode")


class DualSurrogateNoPout:
    def __init__(self, cfg, np_dtype, torch_dtype, device):
        self.cfg = cfg
        self.np_dtype = np_dtype
        self.torch_dtype = torch_dtype
        self.device = device

        self.x_scaler_raw = joblib.load(cfg["x_scaler_path"])
        self.y_scaler_eff_raw = joblib.load(cfg["y_scaler_eff_path"])
        self.y_scaler_aux_raw = joblib.load(cfg["y_scaler_aux_path"])

        self.x_scaler_torch = TorchScaler(self.x_scaler_raw, device, torch_dtype)
        self.y_scaler_eff_torch = TorchScaler(self.y_scaler_eff_raw, device, torch_dtype)
        self.y_scaler_aux_torch = TorchScaler(self.y_scaler_aux_raw, device, torch_dtype)

        self.model_eff = MLPRegressor(
            input_dim=5,
            output_dim=1,
            hidden_dims=cfg["eff_hidden_dims"],
            dropout=cfg["eff_dropout"]
        ).to(device)
        eff_state = safe_torch_load(cfg["model_eff_path"], map_location=device, weights_only=True)
        if isinstance(eff_state, dict) and "model_state_dict" in eff_state:
            eff_state = eff_state["model_state_dict"]
        self.model_eff.load_state_dict(eff_state)
        self.model_eff.eval()

        self.model_aux = MLPRegressor(
            input_dim=5,
            output_dim=3,
            hidden_dims=cfg["aux_hidden_dims"],
            dropout=cfg["aux_dropout"]
        ).to(device)
        aux_state = safe_torch_load(cfg["model_aux_path"], map_location=device, weights_only=True)
        if isinstance(aux_state, dict) and "model_state_dict" in aux_state:
            aux_state = aux_state["model_state_dict"]
        self.model_aux.load_state_dict(aux_state)
        self.model_aux.eval()

        for p in self.model_eff.parameters():
            p.requires_grad_(False)
        for p in self.model_aux.parameters():
            p.requires_grad_(False)

    def predict_batch_np(self, X_raw):
        """
        X_raw: [N,5] = [Vin,Rload,a,b,c]
        返回 eff, vout, irms1, irms2, pout_calc
        """
        X_raw = np.asarray(X_raw, dtype=self.np_dtype)
        X_scaled = self.x_scaler_raw.transform(X_raw).astype(self.np_dtype)
        X_tensor = torch.tensor(X_scaled, dtype=self.torch_dtype, device=self.device)

        with torch.no_grad():
            y_eff_scaled = self.model_eff(X_tensor).detach().cpu().numpy()
            y_aux_scaled = self.model_aux(X_tensor).detach().cpu().numpy()

        eff = self.y_scaler_eff_raw.inverse_transform(y_eff_scaled)[:, 0]
        aux = self.y_scaler_aux_raw.inverse_transform(y_aux_scaled)
        vout = aux[:, 0]
        irms1 = aux[:, 1]
        irms2 = aux[:, 2]
        pout_calc = vout ** 2 / np.maximum(X_raw[:, 1], 1e-12)
        return eff, vout, irms1, irms2, pout_calc

    def forward_torch(self, X_raw_torch):
        """
        可微预测。
        X_raw_torch: [N,5] = [Vin,Rload,a,b,c]
        返回 eff, vout, irms1, irms2, pout_calc
        """
        X_scaled = self.x_scaler_torch.transform(X_raw_torch)
        y_eff_scaled = self.model_eff(X_scaled)
        y_aux_scaled = self.model_aux(X_scaled)

        y_eff = self.y_scaler_eff_torch.inverse_transform(y_eff_scaled)
        y_aux = self.y_scaler_aux_torch.inverse_transform(y_aux_scaled)

        eff = y_eff[:, 0]
        vout = y_aux[:, 0]
        irms1 = y_aux[:, 1]
        irms2 = y_aux[:, 2]
        rload = torch.clamp(X_raw_torch[:, 1], min=1e-12)
        pout_calc = vout ** 2 / rload
        return eff, vout, irms1, irms2, pout_calc



def resolve_column(df, candidates, logical_name):
    """
    Resolve an actual CSV column name from a list of candidate names.

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    candidates : list[str]
        Candidate column names, such as ["V", "Vin"].
    logical_name : str
        Logical variable name used in the program, such as "Vin".
    """
    columns = list(df.columns)

    # 1) Exact match first
    for name in candidates:
        if name in df.columns:
            return name

    # 2) Case-insensitive and whitespace-insensitive match
    normalized_to_actual = {}
    for col in columns:
        key = str(col).strip().lower()
        normalized_to_actual[key] = col

    for name in candidates:
        key = str(name).strip().lower()
        if key in normalized_to_actual:
            return normalized_to_actual[key]

    # 3) More tolerant match: remove spaces, underscores and hyphens
    def compact(s):
        return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    compact_to_actual = {compact(col): col for col in columns}
    for name in candidates:
        key = compact(name)
        if key in compact_to_actual:
            return compact_to_actual[key]

    raise KeyError(
        f"无法为 {logical_name} 找到对应列。候选列={candidates}，CSV实际列={columns}"
    )

def load_input_dataframe(cfg):
    df = pd.read_csv(cfg["csv_path"])

    print("原始 CSV shape =", df.shape)
    print("CSV 列名 =", list(df.columns))

    # 只保留 status=OK 的仿真样本
    status_col = cfg.get("status_column")
    valid_status = cfg.get("valid_status")

    if status_col and status_col in df.columns and valid_status is not None:
        mask = (
            df[status_col]
            .astype(str)
            .str.strip()
            .str.upper()
            == str(valid_status).strip().upper()
        )

        n_before = len(df)
        df = df.loc[mask].copy()

        print(
            f"按 {status_col}={valid_status} 过滤: "
            f"{n_before} -> {len(df)} 条有效样本"
        )

    # 直接按列名映射，不再依赖第几列
    mapping = {}
    for logical_name, candidates in cfg["column_candidates"].items():
        mapping[logical_name] = resolve_column(
            df, candidates, logical_name
        )

    print("\n实际使用的输入列：")
    for logical_name, actual_name in mapping.items():
        print(f"  {logical_name:8s} <- {actual_name}")

    xdf = pd.DataFrame({
        "Vin": df[mapping["Vin"]],
        "Rload": df[mapping["Rload"]],
        "a": df[mapping["a"]],
        "b": df[mapping["b"]],
        "c": df[mapping["c"]],
    })

    for col in xdf.columns:
        xdf[col] = pd.to_numeric(xdf[col], errors="coerce")

    xdf = (
        xdf
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    if len(xdf) == 0:
        raise ValueError("CSV 中没有有效输入数据。")

    print("\nCSV 有效范围：")
    print(f"Vin   : [{xdf['Vin'].min()}, {xdf['Vin'].max()}]")
    print(f"Rload : [{xdf['Rload'].min()}, {xdf['Rload'].max()}]")
    print(f"a     : [{xdf['a'].min()}, {xdf['a'].max()}]")
    print(f"b     : [{xdf['b'].min()}, {xdf['b'].max()}]")
    print(f"c     : [{xdf['c'].min()}, {xdf['c'].max()}]")

    return xdf

def load_abc_bounds_from_csv(xdf, np_dtype):
    bounds = np.array([
        [xdf["a"].min(), xdf["a"].max()],
        [xdf["b"].min(), xdf["b"].max()],
        [xdf["c"].min(), xdf["c"].max()],
    ], dtype=np_dtype)
    print("\n搜索边界（来自 merged_all.csv）:")
    print(f"a: [{bounds[0,0]}, {bounds[0,1]}]")
    print(f"b: [{bounds[1,0]}, {bounds[1,1]}]")
    print(f"c: [{bounds[2,0]}, {bounds[2,1]}]")
    return bounds


def sample_operating_points(cfg, xdf, np_dtype):
    n = int(cfg["n_operating_points"])

    if n <= 0:
        raise ValueError("n_operating_points 必须大于 0。")

    # Vin 范围自动取训练数据中的 V 列范围
    vin_min = float(xdf["Vin"].min())
    vin_max = float(xdf["Vin"].max())

    # Rload 按当前指定范围 100±30 Ω，并限制在训练数据范围内
    r_csv_min = float(xdf["Rload"].min())
    r_csv_max = float(xdf["Rload"].max())

    r_min = max(float(cfg["rload_min"]), r_csv_min)
    r_max = min(float(cfg["rload_max"]), r_csv_max)

    if r_min > r_max:
        raise ValueError(
            f"指定负载范围 [{cfg['rload_min']}, {cfg['rload_max']}] Ω "
            f"与 CSV 负载范围 [{r_csv_min}, {r_csv_max}] Ω 没有交集。"
        )

    if cfg["op_sampling"] == "random_from_csv":
        pairs = (
            xdf[
                (xdf["Rload"] >= r_min)
                & (xdf["Rload"] <= r_max)
            ][["Vin", "Rload"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if len(pairs) == 0:
            raise ValueError("70~130 Ω 范围内没有可用 Vin/Rload 工况。")

        pairs = pairs.sample(
            n=n,
            replace=(len(pairs) < n),
            random_state=cfg["random_seed"]
        ).reset_index(drop=True)

        ops = pairs[["Vin", "Rload"]].values.astype(np_dtype)

    elif cfg["op_sampling"] == "uniform_in_csv_range":
        # Vin 在训练数据范围内连续采样
        # Rload 在 70~130 Ω 内连续采样
        vin = np.random.uniform(vin_min, vin_max, size=n)
        rload = np.random.uniform(r_min, r_max, size=n)
        ops = np.column_stack([vin, rload]).astype(np_dtype)

    else:
        raise ValueError(
            "op_sampling 只能是 random_from_csv 或 uniform_in_csv_range。"
        )

    print("\n本次 NSGA 标签生成范围：")
    print(f"Vin   : [{vin_min}, {vin_max}] V（自动读取 CSV）")
    print(f"Rload : [{r_min}, {r_max}] Ω（100±30 Ω）")
    print(
        f"Vout  : {cfg['Vout_ref']} ± {cfg['Vout_tol_nsga']} V "
        f"= [{cfg['Vout_ref'] - cfg['Vout_tol_nsga']}, "
        f"{cfg['Vout_ref'] + cfg['Vout_tol_nsga']}] V"
    )

    return pd.DataFrame(ops, columns=["Vin", "Rload"])

def quantize_with_step(x, step, lower, upper):
    if step is None or step <= 0:
        return np.clip(x, lower, upper)
    q = np.round((x - lower) / step) * step + lower
    return np.clip(q, lower, upper)


def apply_steps(pop, bounds, cfg):
    pop = np.array(pop, copy=True)
    squeeze_back = False
    if pop.ndim == 1:
        pop = pop.reshape(1, -1)
        squeeze_back = True
    steps = [cfg["a_step"], cfg["b_step"], cfg["c_step"]]
    for j in range(3):
        pop[:, j] = quantize_with_step(pop[:, j], steps[j], bounds[j, 0], bounds[j, 1])
    return pop[0] if squeeze_back else pop


def total_constraint_violation(G):
    return np.sum(np.maximum(G, 0.0), axis=1)


def dominates(F, G, i, j):
    cv_i = np.sum(np.maximum(G[i], 0.0))
    cv_j = np.sum(np.maximum(G[j], 0.0))
    feasible_i = cv_i <= 1e-12
    feasible_j = cv_j <= 1e-12
    if feasible_i and not feasible_j:
        return True
    if not feasible_i and feasible_j:
        return False
    if not feasible_i and not feasible_j:
        return cv_i < cv_j
    no_worse = np.all(F[i] <= F[j] + 1e-12)
    strictly_better = np.any(F[i] < F[j] - 1e-12)
    return no_worse and strictly_better


def fast_non_dominated_sort(F, G):
    N = len(F)
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=int)
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(F, G, p, q):
                S[p].append(q)
            elif dominates(F, G, q, p):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def crowding_distance(F, front):
    if len(front) == 0:
        return np.array([])
    if len(front) <= 2:
        return np.full(len(front), np.inf)
    m = F.shape[1]
    dist = np.zeros(len(front), dtype=np.float64)
    front_F = F[front]
    for obj in range(m):
        order = np.argsort(front_F[:, obj])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        f_min = front_F[order[0], obj]
        f_max = front_F[order[-1], obj]
        if abs(f_max - f_min) < 1e-12:
            continue
        for k in range(1, len(front) - 1):
            dist[order[k]] += (front_F[order[k + 1], obj] - front_F[order[k - 1], obj]) / (f_max - f_min)
    return dist


def build_rank_and_crowding(F, G):
    fronts = fast_non_dominated_sort(F, G)
    rank = np.full(len(F), np.inf)
    crowd = np.zeros(len(F), dtype=np.float64)
    for i, front in enumerate(fronts):
        for idx in front:
            rank[idx] = i
        cd = crowding_distance(F, front)
        for local_k, global_idx in enumerate(front):
            crowd[global_idx] = cd[local_k]
    return fronts, rank, crowd


def tournament_select(pop, rank, crowd):
    i, j = np.random.randint(0, len(pop), size=2)
    if rank[i] < rank[j]:
        return pop[i].copy()
    if rank[j] < rank[i]:
        return pop[j].copy()
    return pop[i].copy() if crowd[i] > crowd[j] else pop[j].copy()


def crossover_blend(p1, p2, crossover_prob):
    if np.random.rand() > crossover_prob:
        return p1.copy(), p2.copy()
    alpha = np.random.rand(3)
    return alpha * p1 + (1.0 - alpha) * p2, alpha * p2 + (1.0 - alpha) * p1


def mutate_gaussian(child, bounds, mutation_prob, mutation_sigma_ratio):
    child = child.copy()
    ranges = bounds[:, 1] - bounds[:, 0]
    sigma = mutation_sigma_ratio * ranges
    for j in range(3):
        if np.random.rand() < mutation_prob:
            child[j] += np.random.randn() * sigma[j]
    return np.clip(child, bounds[:, 0], bounds[:, 1])


def environmental_selection(X, F, G, pred_df, pop_size):
    fronts = fast_non_dominated_sort(F, G)
    new_idx = []
    for front in fronts:
        if len(new_idx) + len(front) <= pop_size:
            new_idx.extend(front)
        else:
            cd = crowding_distance(F, front)
            order = np.argsort(-cd)
            remain = pop_size - len(new_idx)
            new_idx.extend([front[k] for k in order[:remain]])
            break
    new_idx = np.array(new_idx, dtype=int)
    return X[new_idx], F[new_idx], G[new_idx], pred_df.iloc[new_idx].reset_index(drop=True)


def reduce_by_crowding(X, F, G, pred_df, max_size):
    if max_size is None or len(X) <= max_size:
        return X, F, G, pred_df.reset_index(drop=True)
    fronts = fast_non_dominated_sort(F, G)
    keep_idx = []
    for front in fronts:
        if len(keep_idx) + len(front) <= max_size:
            keep_idx.extend(front)
        else:
            remain = max_size - len(keep_idx)
            cd = crowding_distance(F, front)
            order = np.argsort(-cd)
            keep_idx.extend([front[k] for k in order[:remain]])
            break
    keep_idx = np.array(keep_idx, dtype=int)
    return X[keep_idx], F[keep_idx], G[keep_idx], pred_df.iloc[keep_idx].reset_index(drop=True)


class SingleCaseNSGA2NoPout:
    def __init__(self, surrogate, bounds, cfg, vin, rload, np_dtype):
        self.surrogate = surrogate
        self.bounds = bounds.astype(np_dtype)
        self.cfg = cfg
        self.vin = float(vin)
        self.rload = float(rload)
        self.np_dtype = np_dtype
        self.pop_size = int(cfg["nsga_population_size"])
        self.num_generations = int(cfg["nsga_num_generations"])

    def initialize_population(self):
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        pop = np.random.uniform(low=low, high=high, size=(self.pop_size, 3)).astype(self.np_dtype)
        return apply_steps(pop, self.bounds, self.cfg)

    def evaluate_population(self, pop):
        n = len(pop)
        X_full = np.column_stack([
            np.full(n, self.vin, dtype=self.np_dtype),
            np.full(n, self.rload, dtype=self.np_dtype),
            pop[:, 0], pop[:, 1], pop[:, 2]
        ]).astype(self.np_dtype)

        eff, vout, irms1, irms2, pout_calc = self.surrogate.predict_batch_np(X_full)
        irms_obj = np.maximum(irms1, irms2)

        F = np.column_stack([-eff, irms_obj]).astype(self.np_dtype)
        G = np.column_stack([
            np.abs(vout - self.cfg["Vout_ref"]) - self.cfg["Vout_tol_nsga"],
        ]).astype(self.np_dtype)

        cv = total_constraint_violation(G)
        pred_df = pd.DataFrame({
            "Vin": self.vin,
            "Rload": self.rload,
            "a": pop[:, 0],
            "b": pop[:, 1],
            "c": pop[:, 2],
            "efficiency": eff,
            "Vout": vout,
            "Pout_calc": pout_calc,
            "Irms1": irms1,
            "Irms2": irms2,
            "Irms_obj": irms_obj,
            "g_vout": G[:, 0],
            "cv_total": cv,
            "is_feasible": cv <= 1e-12,
        })
        return F, G, pred_df

    def make_offspring(self, pop, rank, crowd):
        children = []
        while len(children) < self.pop_size:
            p1 = tournament_select(pop, rank, crowd)
            p2 = tournament_select(pop, rank, crowd)
            c1, c2 = crossover_blend(p1, p2, self.cfg["nsga_crossover_prob"])
            c1 = mutate_gaussian(c1, self.bounds, self.cfg["nsga_mutation_prob"], self.cfg["nsga_mutation_sigma_ratio"])
            c2 = mutate_gaussian(c2, self.bounds, self.cfg["nsga_mutation_prob"], self.cfg["nsga_mutation_sigma_ratio"])
            c1 = apply_steps(c1, self.bounds, self.cfg)
            c2 = apply_steps(c2, self.bounds, self.cfg)
            children.append(c1)
            if len(children) < self.pop_size:
                children.append(c2)
        return np.array(children, dtype=self.np_dtype)

    def run(self):
        pop = self.initialize_population()
        F, G, pred_df = self.evaluate_population(pop)

        archive_X = pop.copy()
        archive_F = F.copy()
        archive_G = G.copy()
        archive_df = pred_df.copy()

        for _ in range(self.num_generations):
            _, rank, crowd = build_rank_and_crowding(F, G)
            offspring = self.make_offspring(pop, rank, crowd)
            F_off, G_off, df_off = self.evaluate_population(offspring)

            X_comb = np.vstack([pop, offspring])
            F_comb = np.vstack([F, F_off])
            G_comb = np.vstack([G, G_off])
            df_comb = pd.concat([pred_df, df_off], axis=0, ignore_index=True)

            pop, F, G, pred_df = environmental_selection(X_comb, F_comb, G_comb, df_comb, self.pop_size)

            archive_X = np.vstack([archive_X, pop])
            archive_F = np.vstack([archive_F, F])
            archive_G = np.vstack([archive_G, G])
            archive_df = pd.concat([archive_df, pred_df], axis=0, ignore_index=True)

            # Use Python f-strings for archive de-duplication keys.
            # This avoids NumPy string-array addition errors on newer NumPy versions.
            key = np.array([
                f"{row[0]:.8f}_{row[1]:.8f}_{row[2]:.8f}"
                for row in archive_X
            ], dtype=object)
            _, unique_idx = np.unique(key, return_index=True)
            unique_idx = np.sort(unique_idx)
            archive_X = archive_X[unique_idx]
            archive_F = archive_F[unique_idx]
            archive_G = archive_G[unique_idx]
            archive_df = archive_df.iloc[unique_idx].reset_index(drop=True)

            archive_X, archive_F, archive_G, archive_df = reduce_by_crowding(
                archive_X, archive_F, archive_G, archive_df, self.cfg["nsga_archive_max_size"]
            )

        feasible = archive_df["is_feasible"].values.astype(bool)
        if np.any(feasible):
            X_sub = archive_X[feasible]
            F_sub = archive_F[feasible]
            G_sub = archive_G[feasible]
            df_sub = archive_df.loc[feasible].reset_index(drop=True)
        else:
            order = np.argsort(archive_df["cv_total"].values)
            keep = order[:min(self.cfg["nsga_archive_max_size"], len(order))]
            X_sub = archive_X[keep]
            F_sub = archive_F[keep]
            G_sub = archive_G[keep]
            df_sub = archive_df.iloc[keep].reset_index(drop=True)

        fronts = fast_non_dominated_sort(F_sub, G_sub)
        if len(fronts) == 0 or len(fronts[0]) == 0:
            return df_sub

        front_idx = np.array(fronts[0], dtype=int)
        pareto_df = df_sub.iloc[front_idx].reset_index(drop=True)
        pareto_df = pareto_df.sort_values(by=["Irms_obj", "efficiency"], ascending=[True, False]).reset_index(drop=True)
        return pareto_df


def normalize_good_higher(x):
    x = np.asarray(x, dtype=np.float64)
    xmin, xmax = np.min(x), np.max(x)
    if abs(xmax - xmin) < 1e-12:
        return np.ones_like(x)
    return (x - xmin) / (xmax - xmin)


def normalize_good_lower(x):
    x = np.asarray(x, dtype=np.float64)
    xmin, xmax = np.min(x), np.max(x)
    if abs(xmax - xmin) < 1e-12:
        return np.ones_like(x)
    return (xmax - x) / (xmax - xmin)


def select_pareto_point(pareto_df, pref_eff):
    if len(pareto_df) == 0:
        raise ValueError("pareto_df 为空，无法选点。")

    df = pareto_df.copy().reset_index(drop=True)
    pref_eff = float(np.clip(pref_eff, 0.0, 1.0))

    eff_good = normalize_good_higher(df["efficiency"].values)
    irms_good = normalize_good_lower(df["Irms_obj"].values)
    cv = df["cv_total"].values.astype(np.float64)
    cv_bad = normalize_good_higher(cv) if np.max(cv) > 1e-12 else np.zeros_like(cv)

    score = pref_eff * eff_good + (1.0 - pref_eff) * irms_good - 2.0 * cv_bad
    idx = int(np.argmax(score))

    row = df.iloc[idx].copy()
    row["pref_eff"] = pref_eff
    row["pareto_score"] = float(score[idx])
    row["pareto_eff_good"] = float(eff_good[idx])
    row["pareto_irms_good"] = float(irms_good[idx])
    return row



# ============================================================
# 7. 多进程并行生成 NSGA-II 标签
# ============================================================

_WORKER_CONTEXT = {}


def resolve_worker_count(cfg, n_cases):
    """Resolve worker count from CONFIG."""
    requested = cfg.get("n_workers", 1)
    max_workers = cfg.get("max_parallel_workers", None)

    cpu_count = os.cpu_count() or 1

    if isinstance(requested, str) and requested.strip().lower() == "auto":
        n_workers = cpu_count
    else:
        n_workers = int(requested)

    if max_workers is not None:
        n_workers = min(n_workers, int(max_workers))

    n_workers = max(1, min(n_workers, int(n_cases)))
    return n_workers


def init_nsga_worker(cfg, bounds, np_dtype_name):
    """
    Initializer for each worker process.

    Each worker loads one copy of the surrogate models and then handles
    multiple Vin/Rload operating points. This avoids reloading models for
    every single case.
    """
    global _WORKER_CONTEXT

    n_threads = int(cfg.get("torch_num_threads_per_worker", 1))
    n_threads = max(1, n_threads)

    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

    try:
        torch.set_num_threads(n_threads)
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    local_cfg = copy.deepcopy(cfg)

    # In multi-process mode, using one GPU from many processes is usually slower
    # and may cause CUDA memory conflicts. CPU parallelism is more stable here.
    if local_cfg.get("force_cpu_in_parallel", True):
        local_cfg["use_cuda_if_available"] = False

    np_dtype = np.float64 if str(np_dtype_name).lower() == "float64" else np.float32
    np_dtype, torch_dtype, device = get_dtype_and_device(local_cfg)

    set_seed(int(local_cfg.get("random_seed", 42)))

    surrogate = DualSurrogateNoPout(local_cfg, np_dtype, torch_dtype, device)

    _WORKER_CONTEXT = {
        "cfg": local_cfg,
        "bounds": np.asarray(bounds, dtype=np_dtype),
        "np_dtype": np_dtype,
        "surrogate": surrogate,
    }


def run_one_nsga_case_worker(task):
    """Run NSGA-II for one operating point inside a worker process."""
    global _WORKER_CONTEXT

    case_id, vin, rload = task
    case_id = int(case_id)
    vin = float(vin)
    rload = float(rload)

    cfg = _WORKER_CONTEXT["cfg"]
    bounds = _WORKER_CONTEXT["bounds"]
    np_dtype = _WORKER_CONTEXT["np_dtype"]
    surrogate = _WORKER_CONTEXT["surrogate"]

    # Use a deterministic but different seed for every case.
    # This avoids identical random streams across spawned workers.
    seed = int(cfg.get("random_seed", 42)) + case_id * 10007
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    nsga = SingleCaseNSGA2NoPout(
        surrogate=surrogate,
        bounds=bounds,
        cfg=cfg,
        vin=vin,
        rload=rload,
        np_dtype=np_dtype,
    )

    pareto_df = nsga.run().reset_index(drop=True)
    pareto_df["case_id"] = case_id

    labels = []
    for pref in cfg["preference_values"]:
        selected = select_pareto_point(pareto_df, pref)
        selected["case_id"] = case_id
        labels.append(selected)

    labels_df = pd.DataFrame(labels).reset_index(drop=True)

    feasible_count = int(pareto_df["is_feasible"].sum()) if "is_feasible" in pareto_df else 0

    info = {
        "case_id": case_id,
        "Vin": vin,
        "Rload": rload,
        "Pareto": int(len(pareto_df)),
        "Feasible": feasible_count,
    }

    return case_id, labels_df, pareto_df, info


def generate_nsga_preference_labels_parallel(cfg, surrogate_unused, bounds, op_df, np_dtype):
    """
    Parallel version of generate_nsga_preference_labels().

    Parallel unit: one Vin/Rload operating point.
    This is the most natural parallelism because different operating points are
    independent.
    """
    n_cases = len(op_df)
    if n_cases == 0:
        raise ValueError("op_df 为空，无法生成 NSGA 标签。")

    if not cfg.get("parallel_enabled", False):
        return generate_nsga_preference_labels(cfg, surrogate_unused, bounds, op_df, np_dtype)

    n_workers = resolve_worker_count(cfg, n_cases)

    if n_workers <= 1:
        print("Parallel enabled but n_workers <= 1. Use serial mode.")
        return generate_nsga_preference_labels(cfg, surrogate_unused, bounds, op_df, np_dtype)

    print("\n并行 NSGA-II 已启用：")
    print(f"  CPU cores              : {os.cpu_count() or 1}")
    print(f"  n_workers              : {n_workers}")
    print(f"  n_cases                : {n_cases}")
    print(f"  torch threads / worker : {cfg.get('torch_num_threads_per_worker', 1)}")
    print(f"  force CPU in parallel  : {cfg.get('force_cpu_in_parallel', True)}")

    tasks = []
    for i, row in op_df.iterrows():
        case_id = int(i + 1)
        tasks.append((case_id, float(row["Vin"]), float(row["Rload"])))

    np_dtype_name = "float64" if np_dtype == np.float64 else "float32"

    all_label_dfs = []
    all_pareto_dfs = []

    t0 = time.time()
    finished = 0

    # Windows needs spawn context. It is also safe on Linux.
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=init_nsga_worker,
        initargs=(cfg, bounds, np_dtype_name),
    ) as executor:

        future_to_case = {
            executor.submit(run_one_nsga_case_worker, task): task[0]
            for task in tasks
        }

        for future in as_completed(future_to_case):
            case_id = future_to_case[future]

            try:
                case_id, labels_df, pareto_df, info = future.result()
            except Exception as e:
                print(f"[ERROR] NSGA case {case_id} failed: {repr(e)}")
                raise

            all_label_dfs.append(labels_df)
            all_pareto_dfs.append(pareto_df)

            finished += 1
            elapsed = time.time() - t0
            avg_time = elapsed / max(finished, 1)
            remain = avg_time * (n_cases - finished)

            print(
                f"NSGA case {case_id:04d}/{n_cases} finished | "
                f"Vin={info['Vin']:.4f}, Rload={info['Rload']:.4f} | "
                f"Pareto={info['Pareto']}, Feasible={info['Feasible']} | "
                f"Progress={finished}/{n_cases} | "
                f"Elapsed={elapsed/60:.2f} min | "
                f"ETA={remain/60:.2f} min"
            )

    labels_df = pd.concat(all_label_dfs, axis=0, ignore_index=True)
    pareto_all_df = pd.concat(all_pareto_dfs, axis=0, ignore_index=True)

    sort_cols = [c for c in ["case_id", "pref_eff"] if c in labels_df.columns]
    if sort_cols:
        labels_df = labels_df.sort_values(by=sort_cols).reset_index(drop=True)

    if "case_id" in pareto_all_df.columns:
        pareto_all_df = pareto_all_df.sort_values(by=["case_id"]).reset_index(drop=True)

    return labels_df, pareto_all_df

def generate_nsga_preference_labels(cfg, surrogate, bounds, op_df, np_dtype):
    all_labels = []
    all_pareto_rows = []
    t0 = time.time()

    for i, row in op_df.iterrows():
        vin = float(row["Vin"])
        rload = float(row["Rload"])
        nsga = SingleCaseNSGA2NoPout(surrogate, bounds, cfg, vin, rload, np_dtype)
        pareto_df = nsga.run()
        pareto_df["case_id"] = i + 1
        all_pareto_rows.append(pareto_df)

        for pref in cfg["preference_values"]:
            selected = select_pareto_point(pareto_df, pref)
            selected["case_id"] = i + 1
            all_labels.append(selected)

        if cfg["nsga_print_every_case"] and ((i + 1) % cfg["nsga_print_every_case"] == 0):
            feasible_count = int(pareto_df["is_feasible"].sum()) if "is_feasible" in pareto_df else 0
            elapsed = time.time() - t0
            print(
                f"NSGA case {i + 1:04d}/{len(op_df)} | "
                f"Vin={vin:.4f}, Rload={rload:.4f} | "
                f"Pareto={len(pareto_df)}, Feasible={feasible_count} | "
                f"Elapsed={elapsed/60:.2f} min"
            )

    labels_df = pd.DataFrame(all_labels).reset_index(drop=True)
    pareto_all_df = pd.concat(all_pareto_rows, axis=0, ignore_index=True)
    return labels_df, pareto_all_df


def main():
    cfg = CONFIG
    set_seed(cfg["random_seed"])

    # 标签与完整 Pareto 结果统一保存到该目录
    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    ensure_files_exist([
        cfg["csv_path"],
        cfg["model_eff_path"],
        cfg["model_aux_path"],
        cfg["x_scaler_path"],
        cfg["y_scaler_eff_path"],
        cfg["y_scaler_aux_path"],
    ])

    np_dtype, torch_dtype, device = get_dtype_and_device(cfg)
    print("Using device:", device)
    print("Torch dtype:", torch_dtype)

    print("\n加载 CSV 与参数范围...")
    xdf = load_input_dataframe(cfg)
    bounds = load_abc_bounds_from_csv(xdf, np_dtype)

    print("\n采样 NSGA 运行工况...")
    op_df = sample_operating_points(cfg, xdf, np_dtype)
    op_path = os.path.join(save_dir, "operating_points.csv")
    op_df.to_csv(op_path, index=False)

    print(f"运行工况数: {len(op_df)}")
    print(f"每个工况偏好数: {len(cfg['preference_values'])}")
    print(f"预计标签数: {len(op_df) * len(cfg['preference_values'])}")

    print("\n开始逐工况运行 NSGA-II 并生成标签...")

    if cfg.get("parallel_enabled", False):
        # 并行模式下，每个 worker 会各自加载一次代理模型。主进程不再额外加载，节省时间和内存。
        labels_df, pareto_all_df = generate_nsga_preference_labels_parallel(
            cfg, None, bounds, op_df, np_dtype
        )
    else:
        print("\n加载无 Pout 双代理模型...")
        surrogate = DualSurrogateNoPout(cfg, np_dtype, torch_dtype, device)
        labels_df, pareto_all_df = generate_nsga_preference_labels(
            cfg, surrogate, bounds, op_df, np_dtype
        )

    label_path = os.path.join(save_dir, "nsga_pref_labels.csv")
    pareto_path = os.path.join(save_dir, "nsga_all_pareto.csv")

    labels_df.to_csv(label_path, index=False)
    pareto_all_df.to_csv(pareto_path, index=False)

    print("\n================ 标签生成完成 ================")
    print(f"标签数: {len(labels_df)}")
    print(f"标签文件: {os.path.abspath(label_path)}")
    print(f"全部 Pareto 文件: {os.path.abspath(pareto_path)}")
    print(f"工况文件: {os.path.abspath(op_path)}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

