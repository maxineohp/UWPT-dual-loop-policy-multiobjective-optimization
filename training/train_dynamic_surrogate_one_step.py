# -*- coding: utf-8 -*-
"""
train_dynamic_surrogate.py

训练 WPT 动态代理模型。

训练数据目录：
    traj_outputs

输入 CSV 格式：
    time,Vin,Rload,a,b,c,Vout_avg,Pout_avg,eff_avg,Ip_rms,Is_rms

注意：
- eff_avg 不用于训练；
- Pout_avg 不用于训练；
- Pout 后续可由 Vout^2/Rload 计算；
- 代理模型只预测 Vout、Ip_rms、Is_rms 的下一步增量。

动态代理模型形式：
    input =
    [
        Vout(k),
        Ip_rms(k),
        Is_rms(k),

        a(k),
        b(k),
        c(k),

        Δa(k),
        Δb(k),
        Δc(k),

        Vin(k+1),
        Rload(k+1)
    ]

    target =
    [
        Vout(k+1)-Vout(k),
        Ip_rms(k+1)-Ip_rms(k),
        Is_rms(k+1)-Is_rms(k)
    ]

输出文件夹：
    dynamic_surrogate_output/
        dynamic_surrogate.pth
        x_scaler.pkl
        y_scaler.pkl
        train_metrics.csv
        test_metrics.csv
        config.json
"""

import json
import math
import random
from pathlib import Path

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

    "output_dir": "dynamic_surrogate_output",

    "seed": 42,

    # 数据划分按“轨迹文件”划分，避免相邻样本泄漏
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # 网络结构
    "hidden_dims": [256,256,128,64],
    "dropout": 0.0,

    # 训练参数
    "batch_size": 1024,
    "epochs": 1000,
    "lr": 1e-5,
    "weight_decay": 5e-7,
    "patience": 200,

    # 梯度裁剪
    "grad_clip": 5.0,

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
    def __init__(self, input_dim=11, output_dim=3, hidden_dims=None, dropout=0.0):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        layers = []
        last_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h

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


def read_one_csv(path: Path):
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
            df = read_one_csv(path)
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


def evaluate(model, loader, device):
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

    X_train, Y_train, used_train, skipped_train = load_dataset_from_files(train_files, cfg, "train")
    X_val, Y_val, used_val, skipped_val = load_dataset_from_files(val_files, cfg, "val")
    X_test, Y_test, used_test, skipped_test = load_dataset_from_files(test_files, cfg, "test")

    # 标准化
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

    model = DynamicSurrogateMLP(
        input_dim=11,
        output_dim=3,
        hidden_dims=cfg["hidden_dims"],
        dropout=float(cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    wait = 0

    history = []

    best_path = output_dir / "dynamic_surrogate.pth"

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
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

        train_loss = train_loss_sum / max(train_count, 1)

        val_loss, _, _ = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_mse_scaled": train_loss,
            "val_mse_scaled": val_loss,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | train={train_loss:.6e} | val={val_loss:.6e}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "input_dim": 11,
                "output_dim": 3,
                "input_names": [
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
                ],
                "output_names": [
                    "dVout",
                    "dIp_rms",
                    "dIs_rms",
                ],
                "hidden_dims": cfg["hidden_dims"],
            }, best_path)

        else:
            wait += 1

        if wait >= int(cfg["patience"]):
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}, val = {best_val:.6e}")
            break

    # 保存训练曲线
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "train_metrics.csv", index=False)

    # 加载最佳模型
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 测试集评估，注意要反标准化
    test_mse_scaled, y_test_s_true, y_test_s_pred = evaluate(model, test_loader, device)

    y_test_true = y_scaler.inverse_transform(y_test_s_true)
    y_test_pred = y_scaler.inverse_transform(y_test_s_pred)

    metrics_df = compute_metrics_np(
        y_true=y_test_true,
        y_pred=y_test_pred,
        names=["dVout", "dIp_rms", "dIs_rms"],
    )

    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)

    # 保存测试预测散点数据，方便画图
    pred_df = pd.DataFrame({
        "dVout_true": y_test_true[:, 0],
        "dVout_pred": y_test_pred[:, 0],
        "dIp_true": y_test_true[:, 1],
        "dIp_pred": y_test_pred[:, 1],
        "dIs_true": y_test_true[:, 2],
        "dIs_pred": y_test_pred[:, 2],
    })
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # 保存划分信息
    split_info = {
        "train_files": used_train,
        "val_files": used_val,
        "test_files": used_test,
        "skipped_train": skipped_train,
        "skipped_val": skipped_val,
        "skipped_test": skipped_test,
    }

    with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\n训练完成。")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val MSE scaled: {best_val:.6e}")
    print(f"Test MSE scaled: {test_mse_scaled:.6e}")
    print(f"模型保存到: {best_path.resolve()}")
    print(f"测试指标保存到: {(output_dir / 'test_metrics.csv').resolve()}")

    print("\n测试集指标：")
    print(metrics_df)


# ============================================================
# 7. 主入口
# ============================================================

if __name__ == "__main__":
    train_model(CONFIG)