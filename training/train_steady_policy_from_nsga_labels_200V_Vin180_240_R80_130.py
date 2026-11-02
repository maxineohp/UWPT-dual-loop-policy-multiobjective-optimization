# -*- coding: utf-8 -*-
"""读取已有 NSGA-II 标签并训练稳态策略网络；不运行 NSGA-II。"""

import os
import json
import copy
import math
import time
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

    # ---------- 列定义：必须和代理模型训练时一致 ----------
    # 输入: D,E,F,G,H -> 3,4,5,6,7
    "input_idx": [3, 4, 5, 6, 7],
    "input_names": ["Vin", "Rload", "a", "b", "c"],

    # ---------- 代理模型结构：必须和 train_surrogate_no_pout.py 一致 ----------
    "use_float64": True,
    "use_cuda_if_available": True,
    "eff_hidden_dims": [256, 256, 128, 64],
    "eff_dropout": 0.0,
    "aux_hidden_dims": [256, 256, 128, 64],
    "aux_dropout": 0.0,

    # ---------- 稳态控制目标 ----------
    "Vout_ref": 200.0,
    "Vout_tol_nsga": 2.0,
    "Vout_tol_finetune": 1.0,

    # ---------- 本次策略训练使用的工况范围 ----------
    # 只使用 Vin=180~240 V、Rload=80~130 Ω、且满足稳压要求的 NSGA 标签。
    "filter_labels_by_range": True,
    "filter_labels_by_vout": True,
    "train_vin_min": 170.0,
    "train_vin_max": 250.0,
    "train_rload_min": 70.0,
    "train_rload_max": 130.0,

    # ---------- NSGA-II 标签生成 ----------
    "random_seed": 42,
    "n_operating_points": 120,
    "op_sampling": "uniform_in_csv_range",  # uniform_in_csv_range 或 random_from_csv
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

    # a,b,c 是否按固定步长量化；连续优化保持 None
    "a_step": None,
    "b_step": None,
    "c_step": None,

    # ---------- 策略网络结构 ----------
    "policy_hidden_dims": [128, 128, 64, 32],
    "policy_dropout": 0.0,
    "policy_activation": "relu",

    # ---------- 重点工况加权 ----------
    # 让 Vin=150~250 V、Rload=70~130 Ω 区域的样本在监督预训练和自训练中权重更高。
    # 这不会重新运行 NSGA，也不会生成新标签；只改变 loss 中各样本的权重。
    "focus_region_enabled": True,
    "focus_vin_min": 180.0,
    "focus_vin_max": 240.0,
    "focus_rload_min": 80.0,
    "focus_rload_max": 130.0,
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
    "finetune_lr": 3e-5,
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
    "save_dir": "steady_policy_nsga_pref_no_pout_output1",
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


def safe_std(x, eps=1e-12):
    std = np.std(x, axis=0)
    std[std < eps] = 1.0
    return std


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


def load_input_dataframe(cfg):
    df = pd.read_csv(cfg["csv_path"])
    xdf = df.iloc[:, cfg["input_idx"]].copy()
    xdf.columns = cfg["input_names"]
    for col in cfg["input_names"]:
        xdf[col] = pd.to_numeric(xdf[col], errors="coerce")
    xdf = xdf.dropna().reset_index(drop=True)
    if len(xdf) == 0:
        raise ValueError("CSV 中没有有效输入数据。")
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


class SteadyAnglePolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, angle_bounds, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 3))
        self.net = nn.Sequential(*layers)

        bounds = torch.tensor(angle_bounds, dtype=torch.get_default_dtype())
        self.register_buffer("lower", bounds[:, 0].clone())
        self.register_buffer("upper", bounds[:, 1].clone())

    def forward(self, cond_scaled):
        z = self.net(cond_scaled)
        s = torch.sigmoid(z)
        return self.lower + s * (self.upper - self.lower)


class SimpleInputScaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.std[self.std < 1e-12] = 1.0

    def transform_np(self, x):
        return (x - self.mean) / self.std


def split_train_val(df, val_ratio, seed):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def compute_sample_weight(df, cfg):
    """
    对重点工况区域提高训练权重：
        Vin ∈ [focus_vin_min, focus_vin_max]
        Rload ∈ [focus_rload_min, focus_rload_max]

    返回 shape=[N] 的 numpy 数组。
    """
    w_normal = float(cfg.get("normal_sample_weight", 1.0))
    w_focus = float(cfg.get("focus_sample_weight", 1.0))

    weight = np.full(len(df), w_normal, dtype=np.float64)

    if not cfg.get("focus_region_enabled", False):
        return weight

    vin = df["Vin"].values.astype(np.float64)
    rload = df["Rload"].values.astype(np.float64)

    mask = (
        (vin >= float(cfg["focus_vin_min"])) &
        (vin <= float(cfg["focus_vin_max"])) &
        (rload >= float(cfg["focus_rload_min"])) &
        (rload <= float(cfg["focus_rload_max"]))
    )

    weight[mask] = w_focus
    return weight


def make_dataloader(df, input_scaler, cfg, torch_dtype, shuffle):
    cond_raw = df[["Vin", "Rload", "pref_eff"]].values.astype(np.float64)
    cond_scaled = input_scaler.transform_np(cond_raw)
    angle = df[["a", "b", "c"]].values.astype(np.float64)
    sample_weight = compute_sample_weight(df, cfg).astype(np.float64)

    ds = TensorDataset(
        torch.tensor(cond_scaled, dtype=torch_dtype),
        torch.tensor(cond_raw, dtype=torch_dtype),
        torch.tensor(angle, dtype=torch_dtype),
        torch.tensor(sample_weight, dtype=torch_dtype),
    )
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, drop_last=False)


def angle_mse_normalized_per_sample(pred_angle, label_angle, angle_bounds_t):
    span = angle_bounds_t[:, 1] - angle_bounds_t[:, 0]
    span = torch.clamp(span, min=1e-12)
    return torch.mean(((pred_angle - label_angle) / span) ** 2, dim=1)


def weighted_mean(x, sample_weight=None):
    if sample_weight is None:
        return torch.mean(x)
    sample_weight = torch.clamp(sample_weight, min=0.0)
    denom = torch.clamp(torch.sum(sample_weight), min=1e-12)
    return torch.sum(x * sample_weight) / denom


def angle_mse_normalized(pred_angle, label_angle, angle_bounds_t, sample_weight=None):
    per_sample = angle_mse_normalized_per_sample(pred_angle, label_angle, angle_bounds_t)
    return weighted_mean(per_sample, sample_weight)


def evaluate_supervised(policy, loader, angle_bounds_t, device):
    policy.eval()
    losses = []
    with torch.no_grad():
        for cond_scaled, _, angle_label, sample_weight in loader:
            cond_scaled = cond_scaled.to(device)
            angle_label = angle_label.to(device)
            sample_weight = sample_weight.to(device)
            pred = policy(cond_scaled)
            losses.append(float(angle_mse_normalized(pred, angle_label, angle_bounds_t, sample_weight).detach().cpu()))
    return float(np.mean(losses)) if losses else math.inf


def train_supervised(policy, train_loader, val_loader, cfg, device, torch_dtype, angle_bounds):
    angle_bounds_t = torch.tensor(angle_bounds, device=device, dtype=torch_dtype)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["pretrain_lr"], weight_decay=cfg["pretrain_weight_decay"])

    best_state = copy.deepcopy(policy.state_dict())
    best_val = math.inf
    bad_epochs = 0
    hist = []

    for epoch in range(1, cfg["pretrain_epochs"] + 1):
        policy.train()
        train_losses = []
        for cond_scaled, _, angle_label, sample_weight in train_loader:
            cond_scaled = cond_scaled.to(device)
            angle_label = angle_label.to(device)
            sample_weight = sample_weight.to(device)
            opt.zero_grad(set_to_none=True)
            pred = policy(cond_scaled)
            loss = angle_mse_normalized(pred, angle_label, angle_bounds_t, sample_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_supervised(policy, val_loader, angle_bounds_t, device)
        hist.append({"stage": "pretrain", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = copy.deepcopy(policy.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 20 == 0:
            print(f"Pretrain {epoch:04d}/{cfg['pretrain_epochs']} | train={train_loss:.8e} | val={val_loss:.8e}")
        if bad_epochs >= cfg["pretrain_patience"]:
            print(f"Pretrain early stop at epoch {epoch}, best_val={best_val:.8e}")
            break

    policy.load_state_dict(best_state)
    return hist


def auto_eff_divisor(labels_df, cfg):
    if cfg["eff_divisor"] != "auto":
        return float(cfg["eff_divisor"])
    med_eff = float(np.nanmedian(labels_df["efficiency"].values))
    return 100.0 if med_eff > 1.5 else 1.0


def finetune_loss(policy, surrogate, batch, cfg, device, torch_dtype, angle_bounds_t, eff_div, irms_ref, anchor_weight):
    cond_scaled, cond_raw, angle_label, sample_weight = batch
    cond_scaled = cond_scaled.to(device)
    cond_raw = cond_raw.to(device)
    angle_label = angle_label.to(device)
    sample_weight = sample_weight.to(device)

    pred_angle = policy(cond_scaled)
    vin = cond_raw[:, 0]
    rload = cond_raw[:, 1]
    pref = torch.clamp(cond_raw[:, 2], 0.0, 1.0)

    X_raw = torch.column_stack([vin, rload, pred_angle[:, 0], pred_angle[:, 1], pred_angle[:, 2]])
    eff, vout, irms1, irms2, pout_calc = surrogate.forward_torch(X_raw)
    irms_obj = torch.maximum(irms1, irms2)

    vref = torch.tensor(float(cfg["Vout_ref"]), device=device, dtype=torch_dtype)
    v_err_abs = torch.abs(vout - vref)
    v_violation = torch.relu(v_err_abs - float(cfg["Vout_tol_finetune"])) / torch.clamp(vref, min=1e-12)
    loss_vout = v_violation ** 2

    eff_norm = eff / float(eff_div)
    loss_eff = 1.0 - eff_norm

    loss_irms = (irms_obj / max(float(irms_ref), 1e-12)) ** 2

    span = torch.clamp(angle_bounds_t[:, 1] - angle_bounds_t[:, 0], min=1e-12)
    loss_anchor = torch.mean(((pred_angle - angle_label) / span) ** 2, dim=1)

    w_eff = cfg["ft_weight_eff_min"] + pref * (cfg["ft_weight_eff_max"] - cfg["ft_weight_eff_min"])
    w_irms = cfg["ft_weight_irms_min"] + (1.0 - pref) * (cfg["ft_weight_irms_max"] - cfg["ft_weight_irms_min"])

    total = (
        w_eff * loss_eff +
        cfg["ft_weight_vout"] * loss_vout +
        w_irms * loss_irms +
        anchor_weight * loss_anchor
    )

    loss = weighted_mean(total, sample_weight)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "sample_weight_mean": float(torch.mean(sample_weight).detach().cpu()),
        "eff_mean": float(torch.mean(eff).detach().cpu()),
        "vout_mean": float(torch.mean(vout).detach().cpu()),
        "vout_abs_err_mean": float(torch.mean(torch.abs(vout - vref)).detach().cpu()),
        "pout_calc_mean": float(torch.mean(pout_calc).detach().cpu()),
        "irms_obj_mean": float(torch.mean(irms_obj).detach().cpu()),
        "loss_eff": float(torch.mean(loss_eff).detach().cpu()),
        "loss_vout": float(torch.mean(loss_vout).detach().cpu()),
        "loss_irms": float(torch.mean(loss_irms).detach().cpu()),
        "loss_anchor": float(torch.mean(loss_anchor).detach().cpu()),
    }
    return loss, metrics


def evaluate_finetune(policy, surrogate, loader, cfg, device, torch_dtype, angle_bounds_t, eff_div, irms_ref):
    policy.eval()
    metric_sum = {}
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            loss, metrics = finetune_loss(
                policy, surrogate, batch, cfg, device, torch_dtype,
                angle_bounds_t, eff_div, irms_ref, anchor_weight=0.0
            )
            for k, v in metrics.items():
                metric_sum[k] = metric_sum.get(k, 0.0) + v
            n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in metric_sum.items()}


def train_finetune(policy, surrogate, train_loader, val_loader, cfg, labels_df, device, torch_dtype, angle_bounds):
    angle_bounds_t = torch.tensor(angle_bounds, device=device, dtype=torch_dtype)
    eff_div = auto_eff_divisor(labels_df, cfg)
    if cfg["irms_ref"] is None:
        irms_ref = float(np.nanmedian(labels_df["Irms_obj"].values))
        if not np.isfinite(irms_ref) or irms_ref <= 1e-12:
            irms_ref = 1.0
    else:
        irms_ref = float(cfg["irms_ref"])

    print(f"\nFine-tune eff_divisor={eff_div}, irms_ref={irms_ref}")
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["finetune_lr"], weight_decay=cfg["finetune_weight_decay"])

    best_state = copy.deepcopy(policy.state_dict())
    best_val = math.inf
    bad_epochs = 0
    hist = []

    for epoch in range(1, cfg["finetune_epochs"] + 1):
        ratio = (epoch - 1) / max(cfg["finetune_epochs"] - 1, 1)
        anchor_weight = cfg["ft_weight_angle_anchor_start"] * (1.0 - ratio) + cfg["ft_weight_angle_anchor_end"] * ratio

        policy.train()
        train_metric_sum = {}
        n_batches = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, metrics = finetune_loss(policy, surrogate, batch, cfg, device, torch_dtype, angle_bounds_t, eff_div, irms_ref, anchor_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
            opt.step()
            for k, v in metrics.items():
                train_metric_sum[k] = train_metric_sum.get(k, 0.0) + v
            n_batches += 1

        train_metrics = {k: v / max(n_batches, 1) for k, v in train_metric_sum.items()}
        val_metrics = evaluate_finetune(policy, surrogate, val_loader, cfg, device, torch_dtype, angle_bounds_t, eff_div, irms_ref)
        row = {"stage": "finetune", "epoch": epoch, "anchor_weight": anchor_weight}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        hist.append(row)

        val_loss = val_metrics["loss"]
        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = copy.deepcopy(policy.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Finetune {epoch:04d}/{cfg['finetune_epochs']} | "
                f"train_loss={train_metrics.get('loss', math.nan):.8e} | "
                f"val_loss={val_loss:.8e} | "
                f"val_eff={val_metrics.get('eff_mean', math.nan):.6f} | "
                f"val_Verr={val_metrics.get('vout_abs_err_mean', math.nan):.6f} | "
                f"val_Irms={val_metrics.get('irms_obj_mean', math.nan):.6f}"
            )

        if bad_epochs >= cfg["finetune_patience"]:
            print(f"Fine-tune early stop at epoch {epoch}, best_val={best_val:.8e}")
            break

    policy.load_state_dict(best_state)
    return hist, eff_div, irms_ref


def predict_policy_dataframe(policy, surrogate, df, input_scaler, cfg, device, torch_dtype):
    policy.eval()
    cond_raw_np = df[["Vin", "Rload", "pref_eff"]].values.astype(np.float64)
    cond_scaled_np = input_scaler.transform_np(cond_raw_np)

    with torch.no_grad():
        cond_scaled = torch.tensor(cond_scaled_np, device=device, dtype=torch_dtype)
        pred_angle = policy(cond_scaled).detach().cpu().numpy()

    X_raw = np.column_stack([
        cond_raw_np[:, 0],
        cond_raw_np[:, 1],
        pred_angle[:, 0], pred_angle[:, 1], pred_angle[:, 2]
    ])
    eff, vout, irms1, irms2, pout_calc = surrogate.predict_batch_np(X_raw)
    irms_obj = np.maximum(irms1, irms2)

    vref = cfg["Vout_ref"]
    out = pd.DataFrame({
        "Vin": cond_raw_np[:, 0],
        "Rload": cond_raw_np[:, 1],
        "pref_eff": cond_raw_np[:, 2],
        "pred_a": pred_angle[:, 0],
        "pred_b": pred_angle[:, 1],
        "pred_c": pred_angle[:, 2],
        "pred_efficiency": eff,
        "pred_Vout": vout,
        "pred_Pout_calc": pout_calc,
        "pred_Irms1": irms1,
        "pred_Irms2": irms2,
        "pred_Irms_obj": irms_obj,
        "Vout_error": vout - vref,
        "is_voltage_ok": np.abs(vout - vref) <= cfg["Vout_tol_nsga"],
    })

    for col in ["a", "b", "c", "efficiency", "Vout", "Pout_calc", "Irms_obj"]:
        if col in df.columns:
            out[f"label_{col}"] = df[col].values
    return out


def save_plots(save_dir, pre_hist, ft_hist, pred_val_df):
    os.makedirs(save_dir, exist_ok=True)

    pre_df = pd.DataFrame(pre_hist)
    if len(pre_df) > 0:
        plt.figure(figsize=(9, 5))
        plt.plot(pre_df["epoch"], pre_df["train_loss"], label="train")
        plt.plot(pre_df["epoch"], pre_df["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Normalized angle MSE")
        plt.title("Supervised Pretrain Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pretrain_loss_curve.png"), dpi=200)
        plt.close()

    ft_df = pd.DataFrame(ft_hist)
    if len(ft_df) > 0:
        plt.figure(figsize=(9, 5))
        plt.plot(ft_df["epoch"], ft_df["train_loss"], label="train")
        plt.plot(ft_df["epoch"], ft_df["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Fine-tune loss")
        plt.title("Surrogate-based Fine-tune Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "finetune_loss_curve.png"), dpi=200)
        plt.close()

        if "val_eff_mean" in ft_df.columns:
            plt.figure(figsize=(9, 5))
            plt.plot(ft_df["epoch"], ft_df["val_eff_mean"], label="val efficiency")
            plt.xlabel("Epoch")
            plt.ylabel("Efficiency")
            plt.title("Validation Efficiency During Fine-tune")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "finetune_val_eff_curve.png"), dpi=200)
            plt.close()

        if "val_irms_obj_mean" in ft_df.columns:
            plt.figure(figsize=(9, 5))
            plt.plot(ft_df["epoch"], ft_df["val_irms_obj_mean"], label="val Irms_obj")
            plt.xlabel("Epoch")
            plt.ylabel("Irms objective")
            plt.title("Validation Irms During Fine-tune")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "finetune_val_irms_curve.png"), dpi=200)
            plt.close()

    if len(pred_val_df) > 0:
        plt.figure(figsize=(7, 6))
        plt.scatter(pred_val_df["pref_eff"], pred_val_df["pred_efficiency"], s=18, alpha=0.8)
        plt.xlabel("pref_eff")
        plt.ylabel("Predicted efficiency")
        plt.title("Policy Output: Preference vs Efficiency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "val_pref_vs_efficiency.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(7, 6))
        plt.scatter(pred_val_df["pref_eff"], pred_val_df["pred_Irms_obj"], s=18, alpha=0.8)
        plt.xlabel("pref_eff")
        plt.ylabel("Predicted max(Irms1, Irms2)")
        plt.title("Policy Output: Preference vs Irms")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "val_pref_vs_irms.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(7, 6))
        plt.scatter(pred_val_df["pred_Irms_obj"], pred_val_df["pred_efficiency"], s=18, alpha=0.8)
        plt.xlabel("Predicted max(Irms1, Irms2)")
        plt.ylabel("Predicted efficiency")
        plt.title("Policy Validation: Efficiency-Irms Distribution")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "val_eff_irms_scatter.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.hist(pred_val_df["Vout_error"], bins=30)
        plt.xlabel("Vout error")
        plt.ylabel("Count")
        plt.title("Validation Vout Error")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "val_vout_error_hist.png"), dpi=200)
        plt.close()



def filter_and_clean_labels(labels_df, cfg):
    """
    Clean and filter NSGA labels for the requested operating region.

    Current setting:
        Vin   : 150~250 V
        Rload : 70~130 ohm
        Vout  : Vout_ref +/- Vout_tol_nsga
    """
    df = labels_df.copy()

    numeric_cols = [
        "Vin", "Rload", "pref_eff", "a", "b", "c",
        "efficiency", "Vout", "Irms_obj"
    ]
    optional_numeric_cols = [
        "Pout_calc", "Irms1", "Irms2", "g_vout",
        "cv_total", "pareto_score", "pareto_eff_good", "pareto_irms_good"
    ]
    for col in numeric_cols + optional_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n0 = len(df)
    df = (
        df.replace([np.inf, -np.inf], np.nan)
          .dropna(subset=numeric_cols)
          .reset_index(drop=True)
    )
    print(f"标签数值清洗: {n0} -> {len(df)}")

    if cfg.get("filter_labels_by_range", True):
        mask_range = (
            (df["Vin"] >= float(cfg["train_vin_min"])) &
            (df["Vin"] <= float(cfg["train_vin_max"])) &
            (df["Rload"] >= float(cfg["train_rload_min"])) &
            (df["Rload"] <= float(cfg["train_rload_max"]))
        )
        before = len(df)
        df = df.loc[mask_range].copy().reset_index(drop=True)
        print(
            "按工况范围过滤标签: "
            f"{before} -> {len(df)} | "
            f"Vin=[{cfg['train_vin_min']}, {cfg['train_vin_max']}], "
            f"Rload=[{cfg['train_rload_min']}, {cfg['train_rload_max']}]"
        )

    if cfg.get("filter_labels_by_vout", True):
        vref = float(cfg["Vout_ref"])
        vtol = float(cfg["Vout_tol_nsga"])
        mask_v = np.abs(df["Vout"].values.astype(np.float64) - vref) <= vtol
        before = len(df)
        df = df.loc[mask_v].copy().reset_index(drop=True)
        print(
            "按稳压约束过滤标签: "
            f"{before} -> {len(df)} | "
            f"Vout={vref} +/- {vtol} V"
        )

    if len(df) == 0:
        raise ValueError(
            "过滤后 NSGA 标签为空。请确认 nsga_pref_labels.csv 是按 "
            "Vin=150~250 V、Rload=70~130 ohm、Vout=200 V 附近生成的；"
            "或者临时关闭 filter_labels_by_vout / 放宽 Vout_tol_nsga。"
        )

    print("\n训练使用标签范围：")
    print(f"Vin   : [{df['Vin'].min()}, {df['Vin'].max()}]")
    print(f"Rload : [{df['Rload'].min()}, {df['Rload'].max()}]")
    print(f"Vout  : [{df['Vout'].min()}, {df['Vout'].max()}]")
    print(f"a     : [{df['a'].min()}, {df['a'].max()}]")
    print(f"b     : [{df['b'].min()}, {df['b'].max()}]")
    print(f"c     : [{df['c'].min()}, {df['c'].max()}]")

    return df


def main():
    cfg = CONFIG
    set_seed(cfg["random_seed"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    with open(os.path.join(cfg["save_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    ensure_files_exist([
        cfg["csv_path"],
        cfg["model_eff_path"],
        cfg["model_aux_path"],
        cfg["x_scaler_path"],
        cfg["y_scaler_eff_path"],
        cfg["y_scaler_aux_path"],
        cfg["nsga_label_path"],
    ])

    np_dtype, torch_dtype, device = get_dtype_and_device(cfg)
    print("Using device:", device)
    print("Torch dtype:", torch_dtype)

    print("\n加载 CSV 与参数范围...")
    xdf = load_input_dataframe(cfg)
    bounds = load_abc_bounds_from_csv(xdf, np_dtype)

    print("\n加载无 Pout 双代理模型...")
    surrogate = DualSurrogateNoPout(cfg, np_dtype, torch_dtype, device)

    label_path = cfg["nsga_label_path"]
    print(f"\n读取已有 NSGA 标签: {label_path}")
    labels_df = pd.read_csv(label_path)

    if len(labels_df) == 0:
        raise ValueError("NSGA 标签为空，无法训练策略网络。")

    required_cols = ["Vin", "Rload", "pref_eff", "a", "b", "c", "efficiency", "Vout", "Irms_obj"]
    missing = [c for c in required_cols if c not in labels_df.columns]
    if missing:
        raise ValueError(f"NSGA 标签缺少必要列: {missing}")

    print(f"原始标签数: {len(labels_df)}")
    if "is_feasible" in labels_df.columns:
        print(f"原始可行标签比例: {labels_df['is_feasible'].astype(float).mean():.4f}")

    labels_df = filter_and_clean_labels(labels_df, cfg)

    labels_used_path = os.path.join(cfg["save_dir"], "nsga_pref_labels_used.csv")
    labels_df.to_csv(labels_used_path, index=False)

    print(f"最终训练标签数: {len(labels_df)}")
    if "is_feasible" in labels_df.columns:
        print(f"最终可行标签比例: {labels_df['is_feasible'].astype(float).mean():.4f}")

    label_weights = compute_sample_weight(labels_df, cfg)
    focus_mask = label_weights > float(cfg.get("normal_sample_weight", 1.0))
    print(f"重点工况标签数: {int(np.sum(focus_mask))} / {len(labels_df)}")

    train_df, val_df = split_train_val(labels_df, cfg["val_ratio"], cfg["random_seed"])

    cond_train = train_df[["Vin", "Rload", "pref_eff"]].values.astype(np.float64)
    input_scaler = SimpleInputScaler(
        mean=np.mean(cond_train, axis=0),
        std=safe_std(cond_train)
    )

    train_loader = make_dataloader(train_df, input_scaler, cfg, torch_dtype, shuffle=True)
    val_loader = make_dataloader(val_df, input_scaler, cfg, torch_dtype, shuffle=False)

    policy = SteadyAnglePolicy(
        input_dim=3,
        hidden_dims=cfg["policy_hidden_dims"],
        angle_bounds=bounds,
        dropout=cfg["policy_dropout"]
    ).to(device)

    print("\n开始监督预训练：学习 NSGA-II 偏好标签...")
    pre_hist = train_supervised(
        policy, train_loader, val_loader, cfg, device, torch_dtype, bounds
    )

    print("\n开始代理模型引导性能微调...")
    ft_hist, eff_div, irms_ref = train_finetune(
        policy, surrogate, train_loader, val_loader, cfg,
        labels_df, device, torch_dtype, bounds
    )

    print("\n保存模型与验证结果...")
    model_path = os.path.join(cfg["save_dir"], "steady_policy_nsga_pref.pth")
    torch.save({
        "model_state_dict": policy.state_dict(),
        "angle_bounds": bounds,
        "input_dim": 3,
        "hidden_dims": cfg["policy_hidden_dims"],
        "dropout": cfg["policy_dropout"],
        "input_columns": ["Vin", "Rload", "pref_eff"],
        "output_columns": ["a", "b", "c"],
        "eff_divisor": eff_div,
        "irms_ref": irms_ref,
        "config": cfg,
    }, model_path)

    joblib.dump({
        "mean": input_scaler.mean,
        "std": input_scaler.std,
        "columns": ["Vin", "Rload", "pref_eff"],
    }, os.path.join(cfg["save_dir"], "steady_policy_input_scaler.pkl"))

    np.save(os.path.join(cfg["save_dir"], "steady_policy_angle_bounds.npy"), bounds)

    pred_val_df = predict_policy_dataframe(
        policy, surrogate, val_df, input_scaler, cfg, device, torch_dtype
    )
    pred_val_path = os.path.join(cfg["save_dir"], "validation_policy_predictions.csv")
    pred_val_df.to_csv(pred_val_path, index=False)

    hist_df = pd.concat([pd.DataFrame(pre_hist), pd.DataFrame(ft_hist)], axis=0, ignore_index=True)
    hist_path = os.path.join(cfg["save_dir"], "train_history.csv")
    hist_df.to_csv(hist_path, index=False)

    if cfg["save_plots"]:
        save_plots(cfg["save_dir"], pre_hist, ft_hist, pred_val_df)

    summary = {
        "model_path": model_path,
        "source_label_path": label_path,
        "num_labels": int(len(labels_df)),
        "Vout_ref": float(cfg["Vout_ref"]),
        "Vout_tol_nsga": float(cfg["Vout_tol_nsga"]),
        "Vout_tol_finetune": float(cfg["Vout_tol_finetune"]),
        "train_vin_range": [float(cfg["train_vin_min"]), float(cfg["train_vin_max"])],
        "train_rload_range": [float(cfg["train_rload_min"]), float(cfg["train_rload_max"])],
        "validation_voltage_ok_ratio": float(pred_val_df["is_voltage_ok"].mean()),
        "validation_eff_mean": float(pred_val_df["pred_efficiency"].mean()),
        "validation_irms_obj_mean": float(pred_val_df["pred_Irms_obj"].mean()),
        "eff_divisor": float(eff_div),
        "irms_ref": float(irms_ref),
        "policy_hidden_dims": cfg["policy_hidden_dims"],
    }

    with open(os.path.join(cfg["save_dir"], "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ 稳态策略训练完成 ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\npref_eff = 0.0：更偏向降低 Irms")
    print("pref_eff = 0.5：效率与 Irms 折中")
    print("pref_eff = 1.0：更偏向提高效率")


if __name__ == "__main__":
    main()

