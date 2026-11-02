# -*- coding: utf-8 -*-
"""Knowledge distillation of trained WPT steady/dynamic policies into small ReLU MLPs.

Teacher source formats:
  steady_policy_nsga_pref.pth and steady_policy_input_scaler.pkl
  dynamic_policy_best.pth (from train_dynamic_policy_dpc...py)

Outputs are student .pth files in the same checkpoint schema as the original
training scripts. Both exported students are *logit-core* MLPs so that the
existing TI QAT script can process their .net modules and C2000 can apply the
physical output transforms.

IMPORTANT: a 20-degree dynamic teacher is first mapped into its PHYSICAL action,
then CLIPPED to +/-3 degrees and distilled into a student with 3*tanh(logits).
This imitates a clipped teacher, not a newly optimized 3-degree DPC controller.
Validate/retrain against closed-loop trajectories before energizing a WPT stage.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def read_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型文件: {path}")
    # Only load trusted local training checkpoints: PyTorch checkpoint format is pickle-based.
    try:
        result = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        result = torch.load(path, map_location="cpu")
    if not isinstance(result, dict):
        raise TypeError(f"checkpoint 不是字典: {path}")
    return result


def state_dict_from(ck: dict) -> dict:
    for k in ("model_state_dict", "state_dict", "policy_state_dict", "net_state_dict"):
        if k in ck and isinstance(ck[k], dict):
            return ck[k]
    if ck and all(torch.is_tensor(x) for x in ck.values()):
        return ck
    raise ValueError("无法从 checkpoint 解析网络权重 state_dict")


def load_core(ck: dict, description: str):
    """Reconstruct the original trained ReLU MLP's logit core, excluding bounds."""
    sd = state_dict_from(ck)
    linear_weights = []
    for key, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and key.endswith(".weight"):
            m = re.search(r"(?:^|\.)(\d+)\.weight$", key)
            if m:
                linear_weights.append((int(m.group(1)), key, v))
    linear_weights.sort(key=lambda x: x[0])
    if len(linear_weights) < 2:
        raise ValueError(f"{description}: 未找到多层 Linear 权重；请检查 checkpoint 格式")
    dims = [linear_weights[0][2].shape[1]] + [item[2].shape[0] for item in linear_weights]
    if len(linear_weights) != len(set(x[0] for x in linear_weights)):
        raise ValueError("存在重复的网络层编号")
    layers = []
    for i, (_, key, w) in enumerate(linear_weights):
        bkey = key[:-len("weight")] + "bias"
        if bkey not in sd:
            raise KeyError(f"{description}: 缺失偏置: {bkey}")
        lin = nn.Linear(dims[i], dims[i + 1])
        with torch.no_grad():
            lin.weight.copy_(w.to(dtype=torch.float32))
            lin.bias.copy_(sd[bkey].to(dtype=torch.float32))
        layers.append(lin)
        if i < len(linear_weights) - 1:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print(f"{description}: {' -> '.join(map(str, dims))}, "
          f"parameters={sum(p.numel() for p in net.parameters()):,}")
    return net, dims


def get_scaler(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到归一化文件: {path}")
    sc = joblib.load(path)
    if isinstance(sc, dict):
        mean, std = sc["mean"], sc["std"]
    elif hasattr(sc, "mean_") and hasattr(sc, "scale_"):
        mean, std = sc.mean_, sc.scale_
    else:
        raise ValueError(f"无法识别 scaler 文件格式: {path}")
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if np.any(std <= 0) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("scaler mean/std 不合法")
    return mean, std


def get_steady_bounds(ck: dict, npy: Path | None):
    if "angle_bounds" in ck:
        bounds = np.asarray(ck["angle_bounds"], dtype=np.float32)
    elif npy is not None and npy.is_file():
        bounds = np.load(npy).astype(np.float32)
    else:
        raise ValueError("稳态 checkpoint 缺少 angle_bounds，需提供 --steady-bounds")
    if bounds.shape == (2, 3):
        bounds = bounds.T
    if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("稳态角度边界应为 shape=(3,2)，每行 [low,high]")
    return bounds


def make_mlp(dims: list[int]) -> nn.Sequential:
    parts = []
    for j in range(len(dims) - 1):
        parts.append(nn.Linear(dims[j], dims[j + 1]))
        if j != len(dims) - 2:
            parts.append(nn.ReLU())
    return nn.Sequential(*parts)


class SteadyStudent(nn.Module):
    def __init__(self, dims: list[int], bounds: np.ndarray):
        super().__init__()
        self.net = make_mlp(dims)
        self.register_buffer("lower", torch.tensor(bounds[:, 0].copy()))
        self.register_buffer("upper", torch.tensor(bounds[:, 1].copy()))

    def forward(self, normalized: torch.Tensor):
        return self.lower + torch.sigmoid(self.net(normalized)) * (self.upper - self.lower)


class DynamicStudent(nn.Module):
    def __init__(self, dims: list[int], mean: np.ndarray, std: np.ndarray, du_max: float):
        super().__init__()
        self.net = make_mlp(dims)
        self.register_buffer("input_mean", torch.tensor(mean.copy()))
        self.register_buffer("input_std", torch.tensor(std.copy()))
        self.register_buffer("du_max", torch.full((3,), du_max, dtype=torch.float32))

    def forward(self, raw: torch.Tensor):
        logits = self.net((raw - self.input_mean) / self.input_std)
        return self.du_max * torch.tanh(logits)


def make_steady_raw(rng: np.random.Generator, n: int):
    vin = rng.uniform(170.0, 250.0, n)
    rload = rng.uniform(70.0, 130.0, n)
    pref = rng.uniform(0.0, 1.0, n)
    n_main = int(n * 0.70)
    vin[:n_main] = rng.uniform(180.0, 240.0, n_main)
    rload[:n_main] = rng.uniform(80.0, 130.0, n_main)
    return np.column_stack((vin, rload, pref)).astype(np.float32)


def make_dynamic_raw(rng: np.random.Generator, n: int, ck: dict, du_limit: float):
    vref = float(ck.get("Vref", 200.0))
    low = np.asarray(ck.get("angle_low", [90, 90, 90]), dtype=np.float32)
    high = np.asarray(ck.get("angle_high", [180, 180, 150]), dtype=np.float32)
    vout = rng.uniform(vref - 55, vref + 60, n)
    ip = rng.uniform(0.1, 5.6, n)
    is_ = rng.uniform(0.1, 5.1, n)
    integ = rng.uniform(-0.5, 0.5, n)
    angles = rng.uniform(low, high, size=(n, 3))
    target = np.clip(angles + rng.normal(0.0, [12, 12, 9], (n, 3)), low, high)
    # Distillation is for a 3-degree deployment, so previous applied du is also <= 3.
    prev = rng.uniform(-du_limit, du_limit, (n, 3))
    vin = rng.uniform(180, 250, n)
    rload = rng.uniform(80, 130, n)
    return np.column_stack((vout, ip, is_, vref - vout, integ, angles, target,
                            target - angles, prev, vin, rload, np.full(n, vref))).astype(np.float32)


@torch.no_grad()
def steady_teacher_angle(teacher: nn.Sequential, scaled: np.ndarray, bounds: np.ndarray,
                         batch_size: int = 4096) -> np.ndarray:
    chunks = []
    lo = torch.tensor(bounds[:, 0]); hi = torch.tensor(bounds[:, 1])
    for part in np.array_split(scaled, max(1, (len(scaled) + batch_size - 1) // batch_size)):
        z = teacher(torch.from_numpy(np.ascontiguousarray(part, dtype=np.float32)))
        chunks.append((lo + torch.sigmoid(z) * (hi - lo)).numpy())
    return np.concatenate(chunks, axis=0)


def optional_trajectory_data(traj_dir: str, max_rows: int, rng: np.random.Generator,
                             steady_teacher: nn.Sequential, steady_mean: np.ndarray,
                             steady_std: np.ndarray, bounds: np.ndarray, dyn_ck: dict,
                             pref: float, du_limit: float) -> np.ndarray:
    """Read trajectory data from a CSV/XLSX file or from a directory of CSV files.

    If a directory is supplied, CSV files are preferred (so the original per-run
    files are used). If there are no CSV files, XLSX files are considered.
    If an explicit .xlsx file is supplied, that workbook is read directly.

    If error-integral was not measured, use 0. This does NOT reconstruct historical
    closed-loop error-integral state. Use original logged integral if available.
    """
    import pandas as pd
    root = Path(traj_dir)
    if root.is_file():
        if root.suffix.lower() not in (".csv", ".xlsx", ".xls"):
            raise ValueError(f"轨迹文件只支持 CSV/XLSX/XLS：{root}")
        files = [root]
    elif root.is_dir():
        files = sorted(root.glob("*.csv"))
        if not files:
            files = sorted(
                f for pattern in ("*.xlsx", "*.xls")
                for f in root.glob(pattern)
                if not f.name.startswith("~$")
            )
        if not files:
            raise FileNotFoundError(
                f"目录中未找到 CSV 或 Excel 轨迹文件：{root}")
    else:
        raise FileNotFoundError(f"轨迹路径不存在：{root}")

    fields = ["Vin", "Rload", "a", "b", "c", "Vout_avg", "Ip_rms", "Is_rms"]
    chunks = []
    for f in files:
        try:
            if f.suffix.lower() == ".csv":
                sheets = [(f.name, pd.read_csv(f))]
            else:
                # Read all worksheets; summary sheets without required columns are skipped.
                workbook = pd.read_excel(f, sheet_name=None)
                sheets = [(f"{f.name}[{sheet_name}]", df)
                          for sheet_name, df in workbook.items()]
            for sheet_label, df in sheets:
                # Normalize header whitespace, while preserving the original column spelling.
                df.columns = [str(c).strip() for c in df.columns]
                if not all(col in df.columns for col in fields):
                    missing = [col for col in fields if col not in df.columns]
                    print(f"跳过 {sheet_label}：缺少必要列 {missing}")
                    continue
                cols = fields + (["voltage_error_integral"] if "voltage_error_integral" in df.columns else [])
                df = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
                df = df[(df.Vin.between(180, 250)) & (df.Rload.between(80, 130))]
                if len(df):
                    chunks.append(df)
                    print(f"读取轨迹 {sheet_label}: 有效行 {len(df)}")
        except Exception as e:
            print(f"跳过无法读取的轨迹 {f.name}: {e}")
    if not chunks:
        raise ValueError(
            "没有读到有效轨迹数据。需要列名：Vin、Rload、a、b、c、"
            "Vout_avg、Ip_rms、Is_rms。请核对 Excel/CSV 的表头。")
    frame = pd.concat(chunks, ignore_index=True)
    if len(frame) > max_rows:
        sel = rng.choice(len(frame), size=max_rows, replace=False)
        frame = frame.iloc[sel].reset_index(drop=True)
    raw = frame[fields].to_numpy(dtype=np.float32)
    vin, rload = raw[:, 0], raw[:, 1]
    state = raw[:, 5:8]
    angles = raw[:, 2:5]
    target_input = np.column_stack((vin, rload, np.full(len(raw), pref, dtype=np.float32)))
    target = steady_teacher_angle(steady_teacher,
                                  ((target_input - steady_mean) / steady_std).astype(np.float32), bounds)
    low = np.asarray(dyn_ck.get("angle_low", [90,90,90]), dtype=np.float32)
    high = np.asarray(dyn_ck.get("angle_high", [180,180,150]), dtype=np.float32)
    target = np.clip(target, low, high)
    angles = np.clip(angles, low, high)
    # Previous action is not recoverable if only sampled rows are present: use zero.
    prev = np.zeros((len(raw), 3), dtype=np.float32)
    vref = float(dyn_ck.get("Vref", 200.0))
    integral = (frame["voltage_error_integral"].to_numpy(dtype=np.float32)
                if "voltage_error_integral" in frame else np.zeros(len(raw), dtype=np.float32))
    integral = np.clip(integral, -0.5, 0.5)
    data = np.column_stack((state, vref - state[:, 0], integral, angles, target,
                            target - angles, prev, vin, rload,
                            np.full(len(raw), vref))).astype(np.float32)
    print(f"轨迹样本有效行数: {len(data)} (未提供 prev_du 时用0；不推断历史增量)")
    return data


@torch.no_grad()
def build_teacher_targets(teacher: nn.Sequential, normalized: np.ndarray,
                          mode: str, bounds: np.ndarray | None, teacher_du: np.ndarray | None,
                          out_du: float, batch_size: int):
    ans = []
    for start in range(0, len(normalized), batch_size):
        xx = torch.from_numpy(np.ascontiguousarray(normalized[start:start+batch_size]))
        raw = teacher(xx)
        if mode == "steady":
            lo = torch.tensor(bounds[:, 0]); hi = torch.tensor(bounds[:, 1])
            phys = lo + torch.sigmoid(raw) * (hi - lo)
        else:
            scale = torch.tensor(teacher_du)
            original_action = scale * torch.tanh(raw)
            phys = torch.clamp(original_action, -out_du, out_du)
        ans.append(phys.numpy())
    return np.concatenate(ans, axis=0).astype(np.float32)


def train_student(model: nn.Module, X: np.ndarray, Y: np.ndarray, mode: str,
                  train_epochs: int, batch_size: int, lr: float, seed: int,
                  device: torch.device):
    if len(X) < 100:
        raise ValueError("训练样本不足")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    nval = max(32, round(len(X) * 0.15))
    trainidx, validx = idx[nval:], idx[:nval]
    dataset = TensorDataset(torch.from_numpy(X[trainidx]), torch.from_numpy(Y[trainidx]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    Xv = torch.from_numpy(X[validx]).to(device)
    Yv = torch.from_numpy(Y[validx]).to(device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    best_state, best_val = None, float("inf")
    best_epoch, patience = 0, max(12, min(45, train_epochs // 3))
    history = []

    def calc(pred, truth):
        if mode == "steady":
            # Normalize angles by their legal span; units cancel out.
            span = torch.clamp(model.upper - model.lower, min=1.0)
            return ((pred - truth) / span).square().mean()
        return ((pred - truth) / model.du_max).square().mean()

    for epoch in range(1, train_epochs + 1):
        model.train(); train_sum = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = calc(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_sum += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            vals = []
            for xs, ys in zip(Xv.split(batch_size), Yv.split(batch_size)):
                vals.append((float(calc(model(xs), ys).item()), len(xs)))
        val = sum(a * b for a, b in vals) / len(Xv)
        avg = train_sum / len(trainidx)
        history.append(dict(epoch=epoch, train_loss=avg, val_loss=val))
        if val < best_val - 1e-8:
            best_val = val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == train_epochs:
            print(f"{mode:7s} epoch {epoch:3d}/{train_epochs}, "
                  f"train={avg:.6e}, val={val:.6e}")
        if epoch - best_epoch >= patience:
            print(f"{mode}: early stop at epoch {epoch}, best={best_epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    errors = []
    with torch.no_grad():
        for xs, ys in zip(Xv.split(batch_size), Yv.split(batch_size)):
            errors.append((model(xs) - ys).abs().cpu().numpy())
    diffs = np.concatenate(errors).reshape(-1)
    report = dict(train_rows=int(len(trainidx)), val_rows=int(len(validx)),
                  best_epoch=best_epoch, best_val_mse_normalized=best_val,
                  val_abs_mean_degree=float(np.mean(diffs)),
                  val_abs_p95_degree=float(np.percentile(diffs, 95)),
                  val_abs_max_degree=float(np.max(diffs)),
                  student_params=int(sum(p.numel() for p in model.net.parameters())))
    print(mode, "验证角度/增量误差(度): MAE=%.5f, P95=%.5f, MAX=%.5f" %
          (report["val_abs_mean_degree"], report["val_abs_p95_degree"], report["val_abs_max_degree"]))
    return model.cpu(), report, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steady-pth", required=True, type=Path,
                   help="旧稳态模型 steady_policy_nsga_pref.pth")
    p.add_argument("--steady-scaler", required=True, type=Path,
                   help="旧稳态输入归一化 steady_policy_input_scaler.pkl")
    p.add_argument("--steady-bounds", type=Path, default=None,
                   help="仅当旧稳态checkpoint内没有 angle_bounds 时需要")
    p.add_argument("--dynamic-pth", required=True, type=Path,
                   help="旧动态模型 dynamic_policy_best.pth")
    p.add_argument("--out-dir", type=Path, default=Path("lightweight_models"))
    p.add_argument("--mode", choices=["steady", "dynamic", "both"], default="both")
    p.add_argument("--steady-hidden", type=int, nargs="+", default=[32, 32, 16])
    p.add_argument("--dynamic-hidden", type=int, nargs="+", default=[64, 64, 32])
    p.add_argument("--du-limit", type=float, default=3.0)
    p.add_argument("--samples", type=int, default=30000, help="每个模型生成的合成样本数")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pref", type=float, default=0.8,
                   help="用轨迹状态生成稳态目标角时所用效率偏好")
    p.add_argument("--traj-dir", "--traj-file", dest="traj_dir", type=str, default=None,
                   help="可选：轨迹CSV目录，或单个CSV/XLSX/XLS轨迹文件路径")
    p.add_argument("--max-traj-rows", type=int, default=15000)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.du_limit <= 0 or args.samples < 100 or not 0 <= args.pref <= 1:
        p.error("参数 du-limit、samples 或 pref 非法")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = torch.device("cuda" if (args.device == "cuda" or
                       (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU，请设置 --device cpu")
    rng = np.random.default_rng(args.seed)
    sc = read_checkpoint(args.steady_pth)
    dc = read_checkpoint(args.dynamic_pth)
    steady_teacher, sdims = load_core(sc, "steady-teacher")
    dynamic_teacher, ddims = load_core(dc, "dynamic-teacher")
    if sdims[0] != 3 or sdims[-1] != 3 or ddims[0] != 20 or ddims[-1] != 3:
        raise ValueError(f"教师输入输出维度与提供的训练脚本不符: {sdims}, {ddims}")
    mean_s, std_s = get_scaler(args.steady_scaler)
    bounds = get_steady_bounds(sc, args.steady_bounds)
    mean_d = np.asarray(dc.get("input_mean", state_dict_from(dc).get("input_mean")), dtype=np.float32)
    std_d = np.asarray(dc.get("input_std", state_dict_from(dc).get("input_std")), dtype=np.float32)
    old_du = np.asarray(dc.get("du_max", state_dict_from(dc).get("du_max")), dtype=np.float32).reshape(-1)
    if old_du.size == 1:
        old_du = np.repeat(old_du, 3)
    if mean_s.size != 3 or mean_d.size != 20 or std_d.size != 20 or old_du.size != 3:
        raise ValueError("教师 scaler 维度不一致或动态 checkpoint 缺少 du_max")
    if np.any(std_d <= 0) or np.any(old_du <= 0):
        raise ValueError("动态归一化参数或教师 du_max 非法")
    print("Teacher du_max:", old_du, "→ student du_max:", [args.du_limit]*3)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    if args.mode in ("steady", "both"):
        raw = make_steady_raw(rng, args.samples)
        normalized = ((raw - mean_s) / std_s).astype(np.float32)
        targets = build_teacher_targets(steady_teacher, normalized, "steady", bounds,
                                        None, args.du_limit, args.batch_size)
        stu = SteadyStudent([3] + args.steady_hidden + [3], bounds)
        stu, report, history = train_student(stu, normalized, targets, "steady", args.epochs,
                                             args.batch_size, args.lr, args.seed, dev)
        out = args.out_dir / "steady_policy_nsga_pref_no_pout_output"
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(model_state_dict=stu.state_dict(), angle_bounds=bounds,
                          input_dim=3, hidden_dims=args.steady_hidden, dropout=0.0,
                          input_columns=["Vin", "Rload", "pref_eff"],
                          output_columns=["a", "b", "c"],
                          distilled_from=str(args.steady_pth),
                          target="teacher physical angles", config=dict(policy_hidden_dims=args.steady_hidden))
        torch.save(checkpoint, out / "steady_policy_nsga_pref.pth")
        joblib.dump(dict(mean=mean_s, std=std_s,
                         columns=["Vin","Rload","pref_eff"]),
                    out / "steady_policy_input_scaler.pkl")
        np.save(out / "steady_policy_angle_bounds.npy", bounds)
        (out / "distill_history.json").write_text(json.dumps(history, indent=2), encoding="utf8")
        reports["steady"] = report

    if args.mode in ("dynamic", "both"):
        raw = make_dynamic_raw(rng, args.samples, dc, args.du_limit)
        if args.traj_dir:
            raw_extra = optional_trajectory_data(args.traj_dir, args.max_traj_rows, rng,
                                                steady_teacher, mean_s, std_s, bounds,
                                                dc, args.pref, args.du_limit)
            raw = np.concatenate([raw, raw_extra], axis=0)
        normalized = ((raw - mean_d) / std_d).astype(np.float32)
        targets = build_teacher_targets(dynamic_teacher, normalized, "dynamic", None,
                                        old_du, args.du_limit, args.batch_size)
        ratio = np.mean(np.abs(targets) >= args.du_limit - 1e-4)
        print(f"Clipped teacher target fraction: {ratio:.2%} (near +/-{args.du_limit} deg)")
        stu = DynamicStudent([20] + args.dynamic_hidden + [3], mean_d, std_d, args.du_limit)
        stu, report, history = train_student(stu, raw, targets, "dynamic", args.epochs,
                                             args.batch_size, args.lr, args.seed, dev)
        out = args.out_dir / "dynamic_policy_output"
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(model_state_dict=stu.state_dict(), input_dim=20, output_dim=3,
                          hidden_dims=args.dynamic_hidden, dropout=0.0,
                          input_mean=mean_d, input_std=std_d,
                          du_max=np.full(3, args.du_limit, dtype=np.float32),
                          angle_low=np.asarray(dc.get("angle_low", [90,90,90]), dtype=np.float32),
                          angle_high=np.asarray(dc.get("angle_high", [180,180,150]), dtype=np.float32),
                          Vref=float(dc.get("Vref",200)), Ts_ctrl=float(dc.get("Ts_ctrl",200e-6)),
                          input_names=["Vout","Ip_rms","Is_rms","voltage_error",
                                       "voltage_error_integral","a","b","c",
                                       "a_target","b_target","c_target",
                                       "a_error","b_error","c_error",
                                       "da_prev","db_prev","dc_prev","Vin","Rload","Vref"],
                          output_names=["da","db","dc"],
                          distilled_from=str(args.dynamic_pth),
                          target="clip(teacher_du_max*tanh(teacher_logits), +/-3) in physical deg",
                          config=dict(policy_hidden_dims=args.dynamic_hidden, du_max=[args.du_limit]*3))
        torch.save(checkpoint, out / "dynamic_policy_best.pth")
        joblib.dump(dict(mean=mean_d, std=std_d,
                         columns=checkpoint["input_names"]),
                    out / "dynamic_policy_input_scaler.pkl")
        (out / "distill_history.json").write_text(json.dumps(history, indent=2), encoding="utf8")
        report["teacher_saturation_ratio"] = float(ratio)
        reports["dynamic"] = report

    (args.out_dir / "distill_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf8")
    print("\n训练完成，输出目录:", args.out_dir.resolve())
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print("注意：本脚本仅完成 FP32 学生蒸馏；需另行 QAT、NNC 编译、FP32/INT8 对比和闭环验证。")


if __name__ == "__main__":
    main()
