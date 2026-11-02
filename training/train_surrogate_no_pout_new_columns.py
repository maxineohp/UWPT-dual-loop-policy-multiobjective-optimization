# -*- coding: utf-8 -*-
"""
train_surrogate_no_pout.py

更新版稳态代理模型训练脚本。

核心变化：
1) model_eff: [Vin, Rload, a, b, c] -> [efficiency]
2) model_aux: [Vin, Rload, a, b, c] -> [Vout, Irms1, Irms2]
3) Pout 不再作为神经网络回归输出，而是在后处理中由
       Pout_calc = Vout**2 / Rload
   直接计算。

当前 CSV 预期列：
case_id,status,error_msg,V,R,a,b,c,Vout_avg,Pout_avg,eff_avg_last,Ip_rms_last,Is_rms_last

默认文件结构：
project_dir/
    data/merged_all.csv
    train_surrogate_no_pout.py

输出目录：
    model_output_dual_no_pout/
        model_eff.pth
        model_aux.pth
        x_scaler_shared.pkl
        y_scaler_eff.pkl
        y_scaler_aux.pkl
        surrogate_config.json
        model_eff_history.csv
        model_aux_history.csv
        model_eff_test_predictions.csv
        model_aux_test_predictions.csv
        shared_split_indices.csv
        *.png
"""

import os
import json
import time
import copy
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 0. 配置区
# =========================================================
CONFIG = {
    # ---------- 文件 ----------
    "csv_path": "data/merged_all.csv",
    "save_dir": "./model_output_dual_no_pout",

    # ---------- 当前 merged_all.csv 列定义 ----------
    # 当前表格:
    # case_id,status,error_msg,V,R,a,b,c,Vout_avg,Pout_avg,
    # eff_avg_last,Ip_rms_last,Is_rms_last
    #
    # 不再使用“第几列”的数字索引，直接按列名读取，避免列顺序变化导致读错数据。
    "column_candidates": {
        "Vin": ["V", "Vin"],
        "Rload": ["R", "Rload"],
        "a": ["a"],
        "b": ["b"],
        "c": ["c"],
        "Vout": ["Vout_avg_last", "Vout"],
        "Pout_raw": ["Pout_avg_last", "Pout"],
        "efficiency": ["eff_avg_last", "eff_avg", "efficiency"],
        "Irms1": ["Ip_rms_last", "Ip_rms", "Irms1"],
        "Irms2": ["Is_rms_last", "Is_rms", "Irms2"],
    },

    # 只使用仿真成功的数据。
    "status_column": "status",
    "valid_status": "OK",

    "input_names": ["Vin", "Rload", "a", "b", "c"],
    "eff_output_names": ["efficiency"],
    "aux_output_names": ["Vout", "Irms1", "Irms2"],

    # ---------- 数据划分 ----------
    "test_size": 0.15,
    "val_size": 0.15,
    "random_seed": 42,

    # ---------- 精度 ----------
    "use_float64": True,

    # ---------- DataLoader ----------
    "batch_size": 256,
    "num_workers": 0,
    "use_cuda_if_available": True,

    # ---------- efficiency 模型 ----------
    "eff_hidden_dims": [256, 256, 128, 64],
    "eff_dropout": 0.0,
    "eff_lr": 5e-4,
    "eff_weight_decay": 1e-6,
    "eff_num_epochs": 400,
    "eff_patience": 40,

    # 高效率区加权
    # 如果 eff_avg_last 是 0~1 小数形式，0.90 表示 90%。
    "efficiency_threshold": 0.90,
    "low_eff_weight": 1.0,
    "high_eff_weight": 3.0,

    # ---------- aux 模型 ----------
    "aux_hidden_dims": [256, 256, 128, 64],
    "aux_dropout": 0.0,
    "aux_lr": 1e-3,
    "aux_weight_decay": 1e-6,
    "aux_num_epochs": 400,
    "aux_patience": 40,

    # ---------- 学习率调度 ----------
    "scheduler_factor": 0.5,
    "scheduler_patience": 12,

    # ---------- 画图 ----------
    "save_plots": True,
}


# =========================================================
# 1. 工具函数
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, map_location="cpu", weights_only=None):
    if weights_only is None:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# =========================================================
# 2. 数据集与网络
# =========================================================
class RegressionDataset(Dataset):
    def __init__(self, X, Y, sample_weights=None, dtype=torch.float32):
        self.X = torch.tensor(X, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        if sample_weights is None:
            sample_weights = np.ones(len(X), dtype=np.float64)
        self.W = torch.tensor(sample_weights, dtype=dtype)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.W[idx]


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


# =========================================================
# 3. 损失函数
# =========================================================
def weighted_mse_loss(pred, target, sample_weight):
    per_sample = torch.mean((pred - target) ** 2, dim=1)
    return torch.mean(per_sample * sample_weight)


def plain_mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


# =========================================================
# 4. 数据读取与划分
# =========================================================
def _resolve_column(df, candidates, logical_name):
    """按候选列名自动查找实际列，忽略列名前后空格。"""
    stripped_map = {str(c).strip(): c for c in df.columns}
    for name in candidates:
        if name in stripped_map:
            return stripped_map[name]

    raise KeyError(
        f"找不到 {logical_name} 对应列。候选列名={candidates}\n"
        f"当前 CSV 列名={list(df.columns)}"
    )


def load_data(cfg):
    df = pd.read_csv(cfg["csv_path"])
    print("原始 df.shape =", df.shape)
    print("\n当前 CSV 列名：")
    print(list(df.columns))

    # ---------------------------------------------------------
    # 1) 只保留仿真成功数据
    # ---------------------------------------------------------
    status_col = cfg.get("status_column")
    valid_status = cfg.get("valid_status")

    if status_col and status_col in df.columns and valid_status is not None:
        status_text = df[status_col].astype(str).str.strip()
        n_before = len(df)
        df = df.loc[status_text.str.upper() == str(valid_status).strip().upper()].copy()
        print(
            f"\n按 {status_col}={valid_status} 过滤："
            f"{n_before} -> {len(df)} 条有效样本"
        )

        if len(df) == 0:
            raise ValueError(f"过滤后没有 {status_col}={valid_status} 的有效样本。")
    else:
        print("\n未执行 status 过滤。")

    # ---------------------------------------------------------
    # 2) 按列名读取，而不是依赖列的位置
    # ---------------------------------------------------------
    resolved = {}
    for logical_name, candidates in cfg["column_candidates"].items():
        resolved[logical_name] = _resolve_column(
            df, candidates, logical_name
        )

    print("\n实际使用的列映射：")
    for logical_name, actual_name in resolved.items():
        print(f"  {logical_name:10s} <- {actual_name}")

    data = pd.DataFrame({
        "Vin": df[resolved["Vin"]],
        "Rload": df[resolved["Rload"]],
        "a": df[resolved["a"]],
        "b": df[resolved["b"]],
        "c": df[resolved["c"]],
        "efficiency": df[resolved["efficiency"]],
        "Vout": df[resolved["Vout"]],
        "Pout_raw": df[resolved["Pout_raw"]],
        "Irms1": df[resolved["Irms1"]],
        "Irms2": df[resolved["Irms2"]],
    }).copy()

    # ---------------------------------------------------------
    # 3) 转数值并检查非法值
    # ---------------------------------------------------------
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)

    print("\n每列缺失值数量：")
    print(data.isna().sum())

    if data.isna().any().any():
        bad = data[data.isna().any(axis=1)].head()
        print("\n发现异常样本：")
        print(bad)
        raise ValueError(
            "有效样本中仍存在 NaN 或非法值，请检查 merged_all.csv。"
        )

    np_dtype = np.float64 if cfg["use_float64"] else np.float32

    X = data[cfg["input_names"]].values.astype(np_dtype)
    Y_eff = data[["efficiency"]].values.astype(np_dtype)
    Y_aux = data[["Vout", "Irms1", "Irms2"]].values.astype(np_dtype)

    # ---------------------------------------------------------
    # 4) Pout 只用于物理一致性检查，不参与训练
    # ---------------------------------------------------------
    pout_raw = data["Pout_raw"].values.astype(np_dtype)
    rload = np.maximum(
        data["Rload"].values.astype(np_dtype),
        1e-12
    )
    pout_calc_from_true_vout = (
        data["Vout"].values.astype(np_dtype) ** 2 / rload
    )

    pout_check_mae = float(
        np.mean(np.abs(pout_raw - pout_calc_from_true_vout))
    )

    print(
        f"\nPout_raw 与 Vout^2/Rload 的平均绝对差: "
        f"{pout_check_mae:.8f}"
    )

    # 如果 efficiency 数据不是 0~1，给出明确提示。
    eff_median = float(np.median(Y_eff[:, 0]))
    eff_max = float(np.max(Y_eff[:, 0]))
    print(
        f"efficiency 数据范围: "
        f"min={np.min(Y_eff[:, 0]):.8f}, "
        f"median={eff_median:.8f}, "
        f"max={eff_max:.8f}"
    )
    if eff_median > 1.5:
        print(
            "警告：efficiency 看起来可能是 0~100 百分数形式。"
            "如果确实如此，请把 CONFIG['efficiency_threshold'] "
            "从 0.90 改成 90.0。"
        )

    print("\nX shape =", X.shape)
    print("Y_eff shape =", Y_eff.shape)
    print("Y_aux shape =", Y_aux.shape, "  # [Vout, Irms1, Irms2]")

    return X, Y_eff, Y_aux, pout_raw, data


def split_data_shared(X, Y_eff, Y_aux, cfg):
    n = len(X)
    indices = np.arange(n)

    idx_train, idx_temp = train_test_split(
        indices,
        test_size=(cfg["test_size"] + cfg["val_size"]),
        random_state=cfg["random_seed"],
        shuffle=True
    )

    val_ratio_in_temp = cfg["val_size"] / (cfg["test_size"] + cfg["val_size"])
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=(1 - val_ratio_in_temp),
        random_state=cfg["random_seed"],
        shuffle=True
    )

    split = {
        "train": {"X": X[idx_train], "Y_eff": Y_eff[idx_train], "Y_aux": Y_aux[idx_train], "idx": idx_train},
        "val":   {"X": X[idx_val],   "Y_eff": Y_eff[idx_val],   "Y_aux": Y_aux[idx_val],   "idx": idx_val},
        "test":  {"X": X[idx_test],  "Y_eff": Y_eff[idx_test],  "Y_aux": Y_aux[idx_test],  "idx": idx_test},
    }

    print(f"\nTrain samples: {len(idx_train)}")
    print(f"Val samples:   {len(idx_val)}")
    print(f"Test samples:  {len(idx_test)}")
    return split


def compute_eff_sample_weights(Y_eff_raw, threshold=0.90, low_w=1.0, high_w=3.0):
    eff = Y_eff_raw[:, 0]
    weights = np.where(eff >= threshold, high_w, low_w)
    return weights.astype(np.float64 if Y_eff_raw.dtype == np.float64 else np.float32)


# =========================================================
# 5. 推理、训练与评估
# =========================================================
def predict_loader(model, dataloader, device, y_scaler):
    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for xb, yb, _ in dataloader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds_scaled.append(pred)
            trues_scaled.append(yb.numpy())
    preds_scaled = np.vstack(preds_scaled)
    trues_scaled = np.vstack(trues_scaled)
    y_pred = y_scaler.inverse_transform(preds_scaled)
    y_true = y_scaler.inverse_transform(trues_scaled)
    return y_true, y_pred


def compute_loader_loss(model, dataloader, device, use_weight=False):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb, wb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            pred = model(xb)
            loss = weighted_mse_loss(pred, yb, wb) if use_weight else plain_mse_loss(pred, yb)
            losses.append(loss.item())
    return float(np.mean(losses))


def compute_eff_mae_original_scale(model, dataloader, device, y_scaler):
    y_true, y_pred = predict_loader(model, dataloader, device, y_scaler)
    return float(mean_absolute_error(y_true[:, 0], y_pred[:, 0]))


def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    optimizer,
    scheduler,
    y_scaler,
    num_epochs,
    patience,
    save_dir,
    select_metric_name="val_loss",
    use_weight_in_train=False,
    use_weight_in_val=False,
):
    best_state = None
    best_epoch = -1
    best_metric = float("inf")
    wait = 0

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "test_loss": [], "val_select_metric": []}

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        batch_losses = []

        for xb, yb, wb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            pred = model(xb)
            loss = weighted_mse_loss(pred, yb, wb) if use_weight_in_train else plain_mse_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        val_loss = compute_loader_loss(model, val_loader, device, use_weight=use_weight_in_val)
        test_loss = compute_loader_loss(model, test_loader, device, use_weight=False)

        if select_metric_name == "val_eff_mae":
            select_metric = compute_eff_mae_original_scale(model, val_loader, device, y_scaler)
        else:
            select_metric = val_loss

        scheduler.step(select_metric)

        hist["epoch"].append(epoch + 1)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["test_loss"].append(test_loss)
        hist["val_select_metric"].append(select_metric)

        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{model_name}] Epoch {epoch + 1:03d}/{num_epochs} | "
            f"TrainLoss={train_loss:.8f} | ValLoss={val_loss:.8f} | TestLoss={test_loss:.8f} | "
            f"{select_metric_name}={select_metric:.8f} | LR={lr:.2e} | Time={epoch_time:.2f}s"
        )

        if select_metric < best_metric:
            best_metric = select_metric
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n[{model_name}] Early stopping at epoch {epoch + 1}.")
                break

    if best_state is None:
        raise RuntimeError(f"{model_name} 训练失败，没有保存到有效状态。")

    model.load_state_dict(best_state)
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)

    print(f"\n[{model_name}] Best epoch: {best_epoch}")
    print(f"[{model_name}] Best {select_metric_name}: {best_metric:.8f}")
    print(f"[{model_name}] Saved to: {model_path}")

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(os.path.join(save_dir, f"{model_name}_history.csv"), index=False)
    return model, hist_df, model_path


def save_history_plot(hist_df, title, save_path):
    plt.figure(figsize=(9, 5))
    eps = 1e-12
    plt.plot(hist_df["epoch"], np.clip(hist_df["train_loss"].values, eps, None), label="Train Loss")
    plt.plot(hist_df["epoch"], np.clip(hist_df["val_loss"].values, eps, None), label="Val Loss")
    plt.plot(hist_df["epoch"], np.clip(hist_df["test_loss"].values, eps, None), label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.yscale("log")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def print_eff_metrics(y_true, y_pred, title):
    m = regression_metrics(y_true[:, 0], y_pred[:, 0])
    print(f"\n================ {title} ================")
    print(f"efficiency | RMSE = {m['RMSE']:.8f} | MAE = {m['MAE']:.8f} | R2 = {m['R2']:.8f}")


def print_aux_metrics(y_true, y_pred, names, title):
    print(f"\n================ {title} ================")
    for i, name in enumerate(names):
        m = regression_metrics(y_true[:, i], y_pred[:, i])
        print(f"{name:8s} | RMSE = {m['RMSE']:.8f} | MAE = {m['MAE']:.8f} | R2 = {m['R2']:.8f}")


# =========================================================
# 6. 主程序
# =========================================================
def main():
    cfg = CONFIG
    set_seed(cfg["random_seed"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    with open(os.path.join(cfg["save_dir"], "surrogate_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    if cfg["use_float64"]:
        torch.set_default_dtype(torch.float64)
        torch_dtype = torch.float64
        np_dtype = np.float64
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32

    if cfg["use_cuda_if_available"] and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("Torch dtype:", torch_dtype)

    X, Y_eff, Y_aux, pout_raw, df_selected = load_data(cfg)
    split = split_data_shared(X, Y_eff, Y_aux, cfg)

    X_train = split["train"]["X"]
    X_val = split["val"]["X"]
    X_test = split["test"]["X"]

    Y_eff_train = split["train"]["Y_eff"]
    Y_eff_val = split["val"]["Y_eff"]
    Y_eff_test = split["test"]["Y_eff"]

    Y_aux_train = split["train"]["Y_aux"]
    Y_aux_val = split["val"]["Y_aux"]
    Y_aux_test = split["test"]["Y_aux"]

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train).astype(np_dtype)
    X_val_s = x_scaler.transform(X_val).astype(np_dtype)
    X_test_s = x_scaler.transform(X_test).astype(np_dtype)
    joblib.dump(x_scaler, os.path.join(cfg["save_dir"], "x_scaler_shared.pkl"))

    y_eff_scaler = StandardScaler()
    Y_eff_train_s = y_eff_scaler.fit_transform(Y_eff_train).astype(np_dtype)
    Y_eff_val_s = y_eff_scaler.transform(Y_eff_val).astype(np_dtype)
    Y_eff_test_s = y_eff_scaler.transform(Y_eff_test).astype(np_dtype)
    joblib.dump(y_eff_scaler, os.path.join(cfg["save_dir"], "y_scaler_eff.pkl"))

    y_aux_scaler = StandardScaler()
    Y_aux_train_s = y_aux_scaler.fit_transform(Y_aux_train).astype(np_dtype)
    Y_aux_val_s = y_aux_scaler.transform(Y_aux_val).astype(np_dtype)
    Y_aux_test_s = y_aux_scaler.transform(Y_aux_test).astype(np_dtype)
    joblib.dump(y_aux_scaler, os.path.join(cfg["save_dir"], "y_scaler_aux.pkl"))

    eff_train_weights = compute_eff_sample_weights(
        Y_eff_train,
        threshold=cfg["efficiency_threshold"],
        low_w=cfg["low_eff_weight"],
        high_w=cfg["high_eff_weight"],
    ).astype(np_dtype)
    eff_val_weights = np.ones(len(Y_eff_val), dtype=np_dtype)
    eff_test_weights = np.ones(len(Y_eff_test), dtype=np_dtype)

    print(f"\nTrain 高效率区比例: {np.mean(Y_eff_train[:, 0] >= cfg['efficiency_threshold']):.4%}")
    print(f"Val   高效率区比例: {np.mean(Y_eff_val[:, 0] >= cfg['efficiency_threshold']):.4%}")
    print(f"Test  高效率区比例: {np.mean(Y_eff_test[:, 0] >= cfg['efficiency_threshold']):.4%}")

    eff_train_loader = DataLoader(
        RegressionDataset(X_train_s, Y_eff_train_s, eff_train_weights, dtype=torch_dtype),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda")
    )
    eff_val_loader = DataLoader(
        RegressionDataset(X_val_s, Y_eff_val_s, eff_val_weights, dtype=torch_dtype),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda")
    )
    eff_test_loader = DataLoader(
        RegressionDataset(X_test_s, Y_eff_test_s, eff_test_weights, dtype=torch_dtype),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda")
    )

    aux_train_weights = np.ones(len(Y_aux_train), dtype=np_dtype)
    aux_val_weights = np.ones(len(Y_aux_val), dtype=np_dtype)
    aux_test_weights = np.ones(len(Y_aux_test), dtype=np_dtype)

    aux_train_loader = DataLoader(
        RegressionDataset(X_train_s, Y_aux_train_s, aux_train_weights, dtype=torch_dtype),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda")
    )
    aux_val_loader = DataLoader(
        RegressionDataset(X_val_s, Y_aux_val_s, aux_val_weights, dtype=torch_dtype),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda")
    )
    aux_test_loader = DataLoader(
        RegressionDataset(X_test_s, Y_aux_test_s, aux_test_weights, dtype=torch_dtype),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda")
    )

    # ------------------------- model_eff -------------------------
    print("\n" + "=" * 60)
    print("开始训练 model_eff: [Vin,Rload,a,b,c] -> efficiency")
    print("=" * 60)
    model_eff = MLPRegressor(5, 1, cfg["eff_hidden_dims"], cfg["eff_dropout"]).to(device)
    optimizer_eff = torch.optim.AdamW(model_eff.parameters(), lr=cfg["eff_lr"], weight_decay=cfg["eff_weight_decay"])
    scheduler_eff = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_eff, mode="min", factor=cfg["scheduler_factor"], patience=cfg["scheduler_patience"]
    )
    model_eff, eff_hist_df, _ = train_model(
        "model_eff", model_eff, eff_train_loader, eff_val_loader, eff_test_loader, device,
        optimizer_eff, scheduler_eff, y_eff_scaler, cfg["eff_num_epochs"], cfg["eff_patience"], cfg["save_dir"],
        select_metric_name="val_eff_mae", use_weight_in_train=True, use_weight_in_val=False
    )
    if cfg["save_plots"]:
        save_history_plot(eff_hist_df, "model_eff Loss Curve", os.path.join(cfg["save_dir"], "model_eff_loss_curve.png"))

    Y_eff_true_test, Y_eff_pred_test = predict_loader(model_eff, eff_test_loader, device, y_eff_scaler)
    print_eff_metrics(Y_eff_true_test, Y_eff_pred_test, "model_eff Test Metrics")
    pd.DataFrame({
        "true_efficiency": Y_eff_true_test[:, 0],
        "pred_efficiency": Y_eff_pred_test[:, 0],
        "abs_error_efficiency": np.abs(Y_eff_true_test[:, 0] - Y_eff_pred_test[:, 0]),
    }).to_csv(os.path.join(cfg["save_dir"], "model_eff_test_predictions.csv"), index=False)

    # ------------------------- model_aux -------------------------
    print("\n" + "=" * 60)
    print("开始训练 model_aux: [Vin,Rload,a,b,c] -> [Vout, Irms1, Irms2]")
    print("=" * 60)
    model_aux = MLPRegressor(5, 3, cfg["aux_hidden_dims"], cfg["aux_dropout"]).to(device)
    optimizer_aux = torch.optim.AdamW(model_aux.parameters(), lr=cfg["aux_lr"], weight_decay=cfg["aux_weight_decay"])
    scheduler_aux = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_aux, mode="min", factor=cfg["scheduler_factor"], patience=cfg["scheduler_patience"]
    )
    model_aux, aux_hist_df, _ = train_model(
        "model_aux", model_aux, aux_train_loader, aux_val_loader, aux_test_loader, device,
        optimizer_aux, scheduler_aux, y_aux_scaler, cfg["aux_num_epochs"], cfg["aux_patience"], cfg["save_dir"],
        select_metric_name="val_loss", use_weight_in_train=False, use_weight_in_val=False
    )
    if cfg["save_plots"]:
        save_history_plot(aux_hist_df, "model_aux Loss Curve", os.path.join(cfg["save_dir"], "model_aux_loss_curve.png"))

    Y_aux_true_test, Y_aux_pred_test = predict_loader(model_aux, aux_test_loader, device, y_aux_scaler)
    print_aux_metrics(Y_aux_true_test, Y_aux_pred_test, ["Vout", "Irms1", "Irms2"], "model_aux Test Metrics")

    # Pout_calc 从预测 Vout 计算，只做输出记录。
    X_test_raw = X_test
    rload_test = np.maximum(X_test_raw[:, 1], 1e-12)
    true_pout_calc = Y_aux_true_test[:, 0] ** 2 / rload_test
    pred_pout_calc = Y_aux_pred_test[:, 0] ** 2 / rload_test

    aux_pred_df = pd.DataFrame({
        "true_Vout": Y_aux_true_test[:, 0],
        "pred_Vout": Y_aux_pred_test[:, 0],
        "abs_error_Vout": np.abs(Y_aux_true_test[:, 0] - Y_aux_pred_test[:, 0]),
        "true_Irms1": Y_aux_true_test[:, 1],
        "pred_Irms1": Y_aux_pred_test[:, 1],
        "abs_error_Irms1": np.abs(Y_aux_true_test[:, 1] - Y_aux_pred_test[:, 1]),
        "true_Irms2": Y_aux_true_test[:, 2],
        "pred_Irms2": Y_aux_pred_test[:, 2],
        "abs_error_Irms2": np.abs(Y_aux_true_test[:, 2] - Y_aux_pred_test[:, 2]),
        "true_Pout_calc": true_pout_calc,
        "pred_Pout_calc": pred_pout_calc,
        "abs_error_Pout_calc": np.abs(true_pout_calc - pred_pout_calc),
    })
    aux_pred_df.to_csv(os.path.join(cfg["save_dir"], "model_aux_test_predictions.csv"), index=False)

    split_info = pd.DataFrame({
        "train_idx": pd.Series(split["train"]["idx"]),
        "val_idx": pd.Series(split["val"]["idx"]),
        "test_idx": pd.Series(split["test"]["idx"]),
    })
    split_info.to_csv(os.path.join(cfg["save_dir"], "shared_split_indices.csv"), index=False)

    print("\n全部完成，保存目录：", cfg["save_dir"])
    print("已保存内容包括：")
    print("- x_scaler_shared.pkl")
    print("- y_scaler_eff.pkl")
    print("- y_scaler_aux.pkl  # 现在只对应 [Vout, Irms1, Irms2]")
    print("- model_eff.pth")
    print("- model_aux.pth  # output_dim=3")
    print("- model_eff_history.csv / model_aux_history.csv")
    print("- model_eff_test_predictions.csv")
    print("- model_aux_test_predictions.csv")
    print("- shared_split_indices.csv")


if __name__ == "__main__":
    main()
