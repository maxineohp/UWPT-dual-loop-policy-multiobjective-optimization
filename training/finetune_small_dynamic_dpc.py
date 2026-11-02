# -*- coding: utf-8 -*-
"""WPT compact dynamic policy: DPC multi-step fine-tuning after distillation.

Purpose
-------
Continue training the EXISTING small dynamic policy, which was distilled from a
teacher already trained with du_max=[3,3,3]. Use the original frozen dynamic
surrogate for differentiable multi-step control rollouts and add smoothness,
near-equilibrium no-action, and (weak) teacher imitation objectives.

Run from the same directory as:
  lightweight_models/dynamic_policy_output/dynamic_policy_best.pth
  lightweight_models/steady_policy_nsga_pref_no_pout_output/{...}
  dynamic_policy_output6/dynamic_policy_best.pth
  dynamic_surrogate_output/{dynamic_surrogate.pth,x_scaler.pkl,y_scaler.pkl}
  traj_outputs/out_traj_*.csv

Output (does NOT overwrite either teacher or distilled student):
  lightweight_models_dpc/dynamic_policy_output/dynamic_policy_best.pth
  lightweight_models_dpc/dynamic_policy_output/dynamic_policy_input_scaler.pkl
  lightweight_models_dpc/steady_policy_nsga_pref_no_pout_output/(unchanged copy)
  lightweight_models_dpc/dpc_report.json
  lightweight_models_dpc/dpc_history.csv

IMPORTANT: output must be validated in PLECS. The surrogate is an approximation,
so lower surrogate rollout loss is NOT a guarantee of physical closed-loop stability.
No direct hardware output is performed by this script.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURE_NAMES = [
    "Vout", "Ip_rms", "Is_rms", "voltage_error", "voltage_error_integral",
    "a", "b", "c", "a_target", "b_target", "c_target",
    "a_error", "b_error", "c_error", "da_prev", "db_prev", "dc_prev",
    "Vin", "Rload", "Vref",
]
REQUIRED_TRAJ = ["Vin", "Rload", "a", "b", "c", "Vout_avg", "Ip_rms", "Is_rms"]


def load_ckpt(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 checkpoint：{path.resolve()}")
    # .pth is a pickle format; use only checkpoints you produced / trust.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or not isinstance(obj.get("model_state_dict"), dict):
        raise ValueError(f"checkpoint 缺少 model_state_dict：{path}")
    return obj


def load_scaler(path: Path, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 scaler：{path.resolve()}")
    obj = joblib.load(path)
    if isinstance(obj, dict):
        mean = obj.get("mean", obj.get("mean_"))
        std = obj.get("std", obj.get("scale", obj.get("scale_")))
    else:
        mean = getattr(obj, "mean_", None)
        std = getattr(obj, "scale_", None)
    if mean is None or std is None:
        raise ValueError(f"无法解析 mean/std：{path}")
    mean = np.asarray(mean, np.float32).reshape(-1)
    std = np.asarray(std, np.float32).reshape(-1)
    if mean.size != dim or std.size != dim or not np.isfinite(mean).all() or \
            not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"scaler 维度或数值错误: {path}; 期望 {dim}")
    return mean, std


def mlp(dims: List[int]) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i != len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def core_from_ckpt(ck: dict, name: str) -> Tuple[nn.Sequential, List[int]]:
    sd = ck["model_state_dict"]
    weights = []
    for k, w in sd.items():
        match = re.fullmatch(r"net\.(\d+)\.weight", k)
        if match and getattr(w, "ndim", 0) == 2:
            weights.append((int(match.group(1)), k, w))
    weights.sort(key=lambda x: x[0])
    if not weights:
        raise ValueError(f"{name} checkpoint 没有 net.*.weight 层")
    dims = [int(weights[0][2].shape[1])] + [int(w.shape[0]) for _, _, w in weights]
    network = mlp(dims)
    # Only copy neural MLP; ignore input_mean, bounds, du_max buffers in checkpoint.
    network.load_state_dict({k.removeprefix("net."): v for k, v in sd.items()
                             if k.startswith("net.")}, strict=True)
    print(f"{name}: {' -> '.join(map(str, dims))} | 参数量={sum(p.numel() for p in network.parameters()):,}")
    return network, dims


def require_du3(checkpoint: dict, desc: str):
    scale = checkpoint.get("du_max", checkpoint["model_state_dict"].get("du_max"))
    if scale is None:
        raise ValueError(f"{desc} 缺少 du_max，不能确定训练时限幅是否为3°")
    scale = np.asarray(scale, np.float32).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    if scale.size != 3 or not np.allclose(scale, 3.0, atol=1e-4):
        raise ValueError(f"{desc} 原始checkpoint中的 du_max={scale}，并非[3,3,3]。"
                         "请确认是否选错模型，不会擅自缩放教师动作。")


class SmallDynamicPolicy(nn.Module):
    def __init__(self, core: nn.Sequential, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.net = core
        self.register_buffer("input_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("input_std", torch.tensor(std, dtype=torch.float32))
        self.register_buffer("du_max", torch.full((3,), 3.0, dtype=torch.float32))

    def forward(self, raw: torch.Tensor):
        return self.du_max * torch.tanh(self.net((raw - self.input_mean) / self.input_std))


class FrozenSurrogate(nn.Module):
    def __init__(self, ckpt_path: Path, xsc_path: Path, ysc_path: Path):
        super().__init__()
        ck = load_ckpt(ckpt_path)
        core, dims = core_from_ckpt(ck, "frozen dynamic surrogate")
        if dims[0] != 11 or dims[-1] != 3:
            raise ValueError(f"动态代理应为11输入3输出，实际{dims}")
        self.net = core
        xm, xs = load_scaler(xsc_path, 11)
        ym, ys = load_scaler(ysc_path, 3)
        for name, val in (("x_mean",xm),("x_scale",xs),("y_mean",ym),("y_scale",ys)):
            self.register_buffer(name, torch.tensor(val, dtype=torch.float32))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, state, angles, applied_du, condition):
        inp = torch.cat((state, angles, applied_du, condition), dim=1)
        normalized = (inp - self.x_mean) / self.x_scale
        return self.net(normalized) * self.y_scale + self.y_mean


def load_trajectory_groups(path: Path) -> List[Tuple[str, pd.DataFrame]]:
    """Read each CSV as a separate trajectory; Excel each worksheet as a group.

    Prefer CSV files in directory over big merged Excel to avoid losing
    trajectory boundaries / double-counting rows. No cross-file interpolation.
    """
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        files = [f for f in files if f.name.lower().startswith("out_traj") or
                 f.name.lower().startswith("traj_")]
        if not files:
            files = [f for f in sorted(path.glob("*.csv"))]
        if not files:
            raise FileNotFoundError(f"轨迹目录没有CSV文件: {path.resolve()}")
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"轨迹路径不存在: {path.resolve()}")

    groups: List[Tuple[str,pd.DataFrame]] = []
    for f in files:
        try:
            if f.suffix.lower() == ".csv":
                parts = [(f.name, pd.read_csv(f))]
            elif f.suffix.lower() in (".xlsx", ".xls"):
                parts = [(f"{f.name}[{k}]", v) for k, v in
                         pd.read_excel(f, sheet_name=None).items()]
            else:
                continue
            for name, frame in parts:
                frame.columns = [str(col).strip() for col in frame.columns]
                if not set(REQUIRED_TRAJ).issubset(frame.columns):
                    print(f"[跳过] {name} 缺少必要表头 {sorted(set(REQUIRED_TRAJ)-set(frame.columns))}")
                    continue
                use = REQUIRED_TRAJ + (["time"] if "time" in frame.columns else [])
                frame = frame[use].apply(pd.to_numeric, errors="coerce")
                frame = frame.replace([np.inf,-np.inf], np.nan).dropna()
                if "time" in frame:
                    frame = frame.sort_values("time").drop_duplicates("time",keep="last")
                if len(frame) >= 2:
                    groups.append((name, frame.reset_index(drop=True)))
        except Exception as ex:
            print(f"[跳过] 无法读取 {f.name}: {ex}")
    if len(groups) < 2:
        raise ValueError("训练/验证需要至少两条独立轨迹。目前有效轨迹少于2。"
                         "若Excel只有一张合并表且无轨迹标识，请改用原始out_traj_*.csv目录。")
    return groups


def to_scenarios(groups, cfg, rng, max_events, hold_ratio) -> np.ndarray:
    """Return columns [Vout,Ip,Is,a,b,c,Vin_new,Rload_new]."""
    step, hold = [], []
    for _, df in groups:
        ar = df[REQUIRED_TRAJ].to_numpy(dtype=np.float32)
        # [Vin,Rload,a,b,c,Vout,Ip,Is]
        old = ar[:-1]
        new = ar[1:]
        valid = (
            (old[:,0] >= cfg["vin_min"]) & (old[:,0] <= cfg["vin_max"]) &
            (new[:,0] >= cfg["vin_min"]) & (new[:,0] <= cfg["vin_max"]) &
            (old[:,1] >= cfg["r_min"]) & (old[:,1] <= cfg["r_max"]) &
            (new[:,1] >= cfg["r_min"]) & (new[:,1] <= cfg["r_max"])
        )
        cond_changed = (np.abs(new[:,0]-old[:,0]) > cfg["step_v"]) | \
                       (np.abs(new[:,1]-old[:,1]) > cfg["step_r"])
        out = np.column_stack((old[:,5:8], old[:,2:5], new[:,0:2]))
        step.extend(out[valid & cond_changed].tolist())
        hold.extend(out[valid & ~cond_changed].tolist())
    if not step:
        raise ValueError("没有检测到阶跃场景。请检查仿真轨迹的Vin/Rload变化和 --step-v/--step-r。")
    want_step = max(1, int(max_events * (1.0 - hold_ratio)))
    if len(step) > want_step:
        ii = rng.choice(len(step), want_step, replace=False)
        step = [step[i] for i in ii]
    want_hold = min(len(hold), int(len(step) * hold_ratio / max(1e-6,1-hold_ratio)),
                    max_events-len(step))
    if want_hold:
        ii = rng.choice(len(hold), want_hold, replace=False)
        step += [hold[i] for i in ii]
    arr = np.asarray(step, dtype=np.float32).reshape(-1,8)
    rng.shuffle(arr, axis=0)
    return arr


@torch.no_grad()
def fill_targets(scenarios: np.ndarray, steady_net: nn.Module,
                 mean: np.ndarray, std: np.ndarray, angle_bounds: np.ndarray,
                 dyn_low: np.ndarray, dyn_high: np.ndarray,
                 pref: float, vref: float, batch: int, device) -> torch.Tensor:
    if not 0 <= pref <= 1:
        raise ValueError("pref 必须位于[0,1]")
    raw = np.column_stack((scenarios[:,6],scenarios[:,7],
                           np.full(len(scenarios),pref,np.float32)))
    normalized = ((raw-mean)/std).astype(np.float32)
    lows = torch.tensor(angle_bounds[:,0],dtype=torch.float32,device=device)
    highs = torch.tensor(angle_bounds[:,1],dtype=torch.float32,device=device)
    results = []
    for start in range(0,len(normalized),batch):
        xx = torch.tensor(normalized[start:start+batch],device=device)
        z = steady_net(xx)
        results.append((lows + torch.sigmoid(z) * (highs-lows)).cpu())
    target = torch.cat(results,0).numpy()
    target = np.clip(target,dyn_low,dyn_high)
    # x0(3), u0(3), condition(2), target_u(3)
    augmented = np.column_stack((scenarios,target)).astype(np.float32)
    return torch.from_numpy(augmented)


def make_input(state, integral, angles, target, prev, cond, vref):
    error = vref - state[:,0:1]
    return torch.cat((state,error,integral,angles,target,target-angles,prev,
                      cond,vref),dim=1)


def get_bound_array(ck, key, default):
    return np.asarray(ck.get(key, default),dtype=np.float32).reshape(-1)


def loss_rollout(policy, surrogate, teacher, teacher_mean, teacher_std,
                 samples, horizon, hcfg, domain, device, train=True):
    state = samples[:,0:3]
    angles = samples[:,3:6]
    cond = samples[:,6:8]
    target = samples[:,8:11]
    n = len(samples)
    vref = torch.full((n,1),hcfg["vref"],dtype=torch.float32,device=device)
    integral = torch.zeros((n,1),dtype=torch.float32,device=device)
    prev = torch.zeros((n,3),dtype=torch.float32,device=device)
    lo = domain["angle_low"]
    hi = domain["angle_high"]
    state_lo, state_hi = domain["state_low"],domain["state_high"]
    ip_lim, is_lim = domain["ip_lim"],domain["is_lim"]
    scale_ang = domain["angle_scale"]
    ws = hcfg
    total = torch.zeros((),device=device)
    saturation = torch.zeros((),device=device)
    change_sq = torch.zeros((),device=device)
    initial_guide = torch.zeros((),device=device)
    guide_count = 0
    equilibrium_events = torch.zeros((),device=device)

    for k in range(horizon):
        raw = make_input(state,integral,angles,target,prev,cond,vref)
        du = policy(raw)
        next_angle = torch.clamp(angles+du,min=lo,max=hi)
        applied = next_angle-angles
        dstate = surrogate(state,angles,applied,cond)
        next_state_unc = state+dstate
        violation = (torch.relu(state_lo-next_state_unc)/(state_hi-state_lo)).square().mean() + \
                    (torch.relu(next_state_unc-state_hi)/(state_hi-state_lo)).square().mean()
        next_state = torch.clamp(next_state_unc,min=state_lo,max=state_hi)
        ve = (vref-next_state[:,0:1])/ws["volt_scale"]
        ae = (target-next_angle)/scale_ang
        ip_over = (torch.relu(next_state[:,1:2]-ip_lim)/ip_lim).square().mean()
        is_over = (torch.relu(next_state[:,2:3]-is_lim)/is_lim).square().mean()
        curr = (next_state[:,1:2]/ip_lim).square().mean()+\
               (next_state[:,2:3]/is_lim).square().mean()
        action = (applied/3.0).square().mean()
        smooth = ((applied-prev)/3.0).square().mean()

        # Only suppress actions when BOTH voltage error AND target-angle error
        # are already small. No global zero-action bias during transients.
        near = ((vref-state[:,0:1]).abs() <= ws["deadband_v"]) & \
               ((target-angles).abs().amax(dim=1,keepdim=True) <= ws["deadband_angle"])
        near_f = near.float()
        # mean(mask * action^2), not unconditional zero-action penalty.
        hold = (near_f*(du/3.0).square().mean(dim=1,keepdim=True)).mean()
        equilibrium_events += near_f.mean().detach()

        step_loss = (
            ws["w_v"]*ve.square().mean() + ws["w_angle"]*ae.square().mean()
            + ws["w_current_limit"]*(ip_over+is_over)
            + ws["w_current_mag"]*curr
            + ws["w_action"]*action + ws["w_smooth"]*smooth
            + ws["w_hold"]*hold + ws["w_domain"]*violation
            + ws["w_integral"]*integral.square().mean()
        )
        total = total + (ws["gamma"] ** k)*step_loss
        saturation += (du.abs() >= 2.95).float().mean().detach()
        change_sq += smooth.detach()

        if k % 5 == 0:
            # On-policy teacher guidance at rollout states (not only at x0).
            # The teacher itself was trained using +/-3 deg; no reclipping or rescaling.
            with torch.no_grad():
                teacher_du = 3.0*torch.tanh(teacher((raw-teacher_mean)/teacher_std))
            initial_guide = initial_guide + ((du-teacher_du)/3.0).square().mean()
            guide_count += 1

        integral = torch.clamp(integral + ws["dt"]*(vref-next_state[:,0:1]),
                               min=ws["i_min"],max=ws["i_max"])
        prev = applied
        angles = next_angle
        state = next_state

    final_volt_mae = (vref-state[:,0:1]).abs().mean()
    final_angle_mae = (target-angles).abs().mean()
    terminal = ws["w_terminal_v"]*((vref-state[:,0:1])/ws["volt_scale"]).square().mean() + \
               ws["w_terminal_angle"]*((target-angles)/scale_ang).square().mean()
    initial_guide = initial_guide / max(1,guide_count)
    loss = total/horizon + terminal + ws["w_kd"]*initial_guide
    metrics = dict(
        loss=float(loss.detach().item()),
        voltage_mae=float(final_volt_mae.detach().item()),
        angle_mae=float(final_angle_mae.detach().item()),
        saturation_ratio=float((saturation/horizon).item()),
        du_change_rms=float(torch.sqrt(change_sq/horizon).item()*3.0),
        near_equilibrium_ratio=float((equilibrium_events/horizon).item()),
        initial_kd_loss=float(initial_guide.detach().item()),
    )
    return loss,metrics


def mean_metrics(records: List[Tuple[dict,int]]) -> dict:
    n = sum(sz for _, sz in records)
    return {key:float(sum(m[key]*sz for m,sz in records)/max(1,n))
            for key in records[0][0]}


def evaluate(model,surrogate,teacher,tm,ts,loader,horizon,cfg,domain,device):
    model.eval()
    recs=[]
    with torch.no_grad():
        for (x,) in loader:
            x=x.to(device)
            _,met=loss_rollout(model,surrogate,teacher,tm,ts,x,horizon,cfg,domain,device)
            recs.append((met,len(x)))
    return mean_metrics(recs)


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--root",type=Path,default=Path(__file__).resolve().parent,
                    help="工作目录（默认脚本所在文件夹）")
    pa.add_argument("--student-dir",default="lightweight_models")
    pa.add_argument("--teacher-dir",default="dynamic_policy_output6")
    pa.add_argument("--surrogate-dir",default="dynamic_surrogate_output")
    pa.add_argument("--traj-path",default="traj_outputs",
                    help="原始CSV轨迹目录；也可指定包含独立sheet的Excel")
    pa.add_argument("--out-dir",default="lightweight_models_dpc")
    pa.add_argument("--epochs",type=int,default=70)
    pa.add_argument("--batch-size",type=int,default=64)
    pa.add_argument("--lr",type=float,default=3e-5)
    pa.add_argument("--max-scenarios",type=int,default=1800)
    pa.add_argument("--hold-ratio",type=float,default=0.25)
    pa.add_argument("--pref",type=float,default=0.8)
    pa.add_argument("--step-v",type=float,default=3.)
    pa.add_argument("--step-r",type=float,default=1.)
    pa.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    pa.add_argument("--seed",type=int,default=42)
    pa.add_argument("--threads",type=int,default=2,help="CPU训练线程数，小MLP建议1-4")
    # Vibration suppression and DPC weighting. Larger smoothness discourages
    # +/-3 alternating actions, but excessive damping may slow response.
    pa.add_argument("--w-smooth",type=float,default=0.6)
    pa.add_argument("--w-hold",type=float,default=2.0)
    pa.add_argument("--w-kd",type=float,default=0.18)
    pa.add_argument("--deadband-v",type=float,default=2.0)
    pa.add_argument("--deadband-angle",type=float,default=2.0)
    args=pa.parse_args()
    if args.epochs <= 0 or args.batch_size < 1 or args.max_scenarios < 20 or not (0 <= args.hold_ratio < 1):
        pa.error("epochs/batch-size/max-scenarios/hold-ratio 参数不合法")
    if args.threads < 1:
        pa.error("threads 必须 >=1")
    torch.set_num_threads(args.threads)
    rng=np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); random.seed(args.seed)
    device=torch.device("cuda" if (args.device=="cuda" or (args.device=="auto" and
                         torch.cuda.is_available())) else "cpu")
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA GPU，请用 --device cpu")
    print("device:",device)
    root=args.root.resolve()
    student_dir=root/args.student_dir
    teacher_dir=root/args.teacher_dir
    surrogate_dir=root/args.surrogate_dir
    traj_path=root/args.traj_path
    outdir=root/args.out_dir
    if outdir.resolve()==student_dir.resolve():
        raise ValueError("输出目录不能与原学生目录相同，禁止覆盖现有小模型")
    if outdir.resolve()==teacher_dir.resolve():
        raise ValueError("输出目录不能与原教师目录相同")

    # Student checkpoint (from the earlier distillation code).
    stu_file=student_dir/"dynamic_policy_output"/"dynamic_policy_best.pth"
    stu_ck=load_ckpt(stu_file)
    require_du3(stu_ck,"小模型")
    student_core, dims=core_from_ckpt(stu_ck,"existing small dynamic student")
    if dims[0]!=20 or dims[-1]!=3:
        raise ValueError(f"小模型维度异常: {dims}")
    mean_s=np.asarray(stu_ck["input_mean"],np.float32).reshape(-1)
    std_s=np.asarray(stu_ck["input_std"],np.float32).reshape(-1)
    if mean_s.size!=20 or std_s.size!=20 or np.any(std_s<=0):
        raise ValueError("小模型归一化参数不合法")
    policy=SmallDynamicPolicy(student_core,mean_s,std_s).to(device)

    teacher_ck=load_ckpt(teacher_dir/"dynamic_policy_best.pth")
    require_du3(teacher_ck,"原动态教师网络")
    teacher,t_dims=core_from_ckpt(teacher_ck,"original dynamic teacher")
    if t_dims[0]!=20 or t_dims[-1]!=3:
        raise ValueError("教师不是20输入3输出网络")
    tmean=torch.tensor(np.asarray(teacher_ck["input_mean"],np.float32),device=device)
    tstd=torch.tensor(np.asarray(teacher_ck["input_std"],np.float32),device=device)
    if tmean.numel()!=20 or tstd.numel()!=20 or (tstd<=0).any():
        raise ValueError("教师归一化参数不合法")
    teacher=teacher.to(device).eval()
    for p in teacher.parameters():p.requires_grad_(False)

    # Frozen differentiable plant surrogate, trained from simulation.
    surrogate=FrozenSurrogate(surrogate_dir/"dynamic_surrogate.pth",
                              surrogate_dir/"x_scaler.pkl",
                              surrogate_dir/"y_scaler.pkl").to(device)

    # Original STEADY teacher is used only for consistent DPC target angles.
    steady_dir=student_dir/"steady_policy_nsga_pref_no_pout_output"
    steady_ck=load_ckpt(steady_dir/"steady_policy_nsga_pref.pth")
    steady_net,steady_dims=core_from_ckpt(steady_ck,"distilled steady policy (target generator)")
    if steady_dims[0]!=3 or steady_dims[-1]!=3:
        raise ValueError("稳态策略输入/输出不是3维")
    steady_net=steady_net.to(device).eval()
    sm,ss=load_scaler(steady_dir/"steady_policy_input_scaler.pkl",3)
    bounds=np.load(steady_dir/"steady_policy_angle_bounds.npy").astype(np.float32)
    if bounds.shape==(2,3):bounds=bounds.T
    if bounds.shape!=(3,2):
        raise ValueError("稳态angle_bounds必须为(3,2)")

    low=get_bound_array(teacher_ck,"angle_low",[90,90,90])
    high=get_bound_array(teacher_ck,"angle_high",[180,180,150])
    vref=float(teacher_ck.get("Vref",200.0))
    dt=float(teacher_ck.get("Ts_ctrl",200e-6))
    cfg={
        "vin_min":180.,"vin_max":250.,"r_min":80.,"r_max":130.,
        "step_v":args.step_v,"step_r":args.step_r,
    }
    groups=load_trajectory_groups(traj_path)
    rng.shuffle(groups)
    nval=max(1,int(round(0.2*len(groups))))
    train_groups=groups[nval:]; val_groups=groups[:nval]
    if not train_groups or not val_groups:
        raise ValueError("无法按轨迹文件划分训练集、验证集")
    print(f"读取{len(groups)}条有效轨迹；训练{len(train_groups)}条，验证{len(val_groups)}条")
    train_np=to_scenarios(train_groups,cfg,rng,args.max_scenarios,args.hold_ratio)
    val_np=to_scenarios(val_groups,cfg,rng,max(100,args.max_scenarios//3),args.hold_ratio)
    print(f"有效阶跃/保持初始场景: train={len(train_np)}, val={len(val_np)}")
    tr=fill_targets(train_np,steady_net,sm,ss,bounds,low,high,args.pref,vref,1024,device)
    va=fill_targets(val_np,steady_net,sm,ss,bounds,low,high,args.pref,vref,1024,device)
    train_loader=DataLoader(TensorDataset(tr),batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(TensorDataset(va),batch_size=args.batch_size,shuffle=False,num_workers=0)

    # Physical limits / valid region: honor the ORIGINAL dynamic policy runtime
    # if present. Do not quietly change voltage/current hardware limits.
    runtime=teacher_ck.get("runtime",{})
    combined=np.concatenate([train_np[:,:3],val_np[:,:3]],axis=0)
    ip_lim=float(runtime.get("Ip_limit",np.quantile(combined[:,1],0.995)*1.5))
    is_lim=float(runtime.get("Is_limit",np.quantile(combined[:,2],0.995)*1.5))
    s_lo=np.asarray(runtime.get("state_lower",[
        max(0.,float(combined[:,0].min())*0.5),0.,0.]),dtype=np.float32)
    s_hi=np.asarray(runtime.get("state_upper",[
        max(float(combined[:,0].max())*1.5,vref*1.5),
        max(float(combined[:,1].max())*1.5,ip_lim*1.2),
        max(float(combined[:,2].max())*1.5,is_lim*1.2)]),dtype=np.float32)
    if s_lo.shape!=(3,) or s_hi.shape!=(3,) or np.any(s_hi<=s_lo) or ip_lim<=0 or is_lim<=0:
        raise ValueError("代理模型安全范围/电流限值参数异常")
    domain={
        "angle_low":torch.tensor(low,device=device),"angle_high":torch.tensor(high,device=device),
        "angle_scale":torch.tensor([30.,30.,20.],device=device),
        "state_low":torch.tensor(s_lo,device=device),"state_high":torch.tensor(s_hi,device=device),
        "ip_lim":ip_lim,"is_lim":is_lim,
    }
    cp=teacher_ck.get("config",{})
    hcfg=dict(
        vref=vref,dt=dt,i_min=float(cp.get("integral_min",-0.5)),
        i_max=float(cp.get("integral_max",0.5)),volt_scale=5.0,
        deadband_v=args.deadband_v,deadband_angle=args.deadband_angle,
        gamma=0.995,w_v=12.0,w_angle=1.0,w_current_limit=30.0,
        w_current_mag=0.02,w_action=0.10,w_smooth=args.w_smooth,
        w_hold=args.w_hold,w_domain=20.0,w_integral=0.005,
        w_terminal_v=40.0,w_terminal_angle=6.0,w_kd=args.w_kd,
    )
    # Fixed validation horizon so metrics are comparable across epochs.
    # Match the longest 70-step (14 ms at 200 us) original DPC rollout.
    validation_horizon=70
    baseline=evaluate(policy,surrogate,teacher,tmean,tstd,val_loader,
                      validation_horizon,hcfg,domain,device)
    print("微调前验证指标:",json.dumps(baseline,ensure_ascii=False))

    optimizer=torch.optim.AdamW(policy.parameters(),lr=args.lr,weight_decay=1e-6)
    # Never replace the existing student with a worse validation candidate.
    best_val=baseline["loss"]
    best_state=copy.deepcopy(policy.state_dict())
    best_epoch=0
    history=[]
    patience=max(15,int(args.epochs*0.35))
    for epoch in range(1,args.epochs+1):
        # Original DPC used 10/20/50/70 steps; keep a shorter-to-longer
        # curriculum. At Ts=200us the horizons represent 2/4/10/14ms.
        if epoch <= max(1,int(args.epochs*0.20)):
            horizon=10
        elif epoch <= max(2,int(args.epochs*0.45)):
            horizon=20
        elif epoch <= max(3,int(args.epochs*0.70)):
            horizon=50
        else:
            horizon=70
        policy.train()
        records=[]
        for (xx,) in train_loader:
            xx=xx.to(device)
            loss,m=loss_rollout(policy,surrogate,teacher,tmean,tstd,xx,horizon,hcfg,domain,device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Epoch {epoch} DPC loss为NaN/Inf；检查代理模型和学习率")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(),2.0)
            optimizer.step()
            records.append((m,len(xx)))
        trained=mean_metrics(records)
        validated=evaluate(policy,surrogate,teacher,tmean,tstd,val_loader,
                           validation_horizon,hcfg,domain,device)
        history.append(dict(epoch=epoch,horizon=horizon,
                            **{f"train_{k}":v for k,v in trained.items()},
                            **{f"val_{k}":v for k,v in validated.items()}))
        if validated["loss"] < best_val-1e-7:
            best_val=validated["loss"]
            best_state=copy.deepcopy(policy.state_dict())
            best_epoch=epoch
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            print(f"[{epoch:03d}/{args.epochs}] H={horizon} "
                  f"train={trained['loss']:.5f} val={validated['loss']:.5f} "
                  f"Verr={validated['voltage_mae']:.3f}V "
                  f"Aerr={validated['angle_mae']:.3f}deg "
                  f"sat={validated['saturation_ratio']:.1%} "
                  f"dDu_RMS={validated['du_change_rms']:.3f}deg")
        if epoch-best_epoch>=patience:
            print(f"Early stopping: epoch={epoch}, best_epoch={best_epoch}")
            break

    policy.load_state_dict(best_state)
    final=evaluate(policy,surrogate,teacher,tmean,tstd,val_loader,
                   validation_horizon,hcfg,domain,device)
    print("微调后验证指标:",json.dumps(final,ensure_ascii=False))
    outdir.mkdir(parents=True,exist_ok=True)
    dst=outdir/"dynamic_policy_output"
    dst.mkdir(parents=True,exist_ok=True)
    checkpoint=dict(stu_ck)
    checkpoint["model_state_dict"]={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    checkpoint["du_max"]=np.array([3.0,3.0,3.0],dtype=np.float32)
    checkpoint["input_mean"]=mean_s.copy()
    checkpoint["input_std"]=std_s.copy()
    checkpoint["hidden_dims"]=dims[1:-1]
    checkpoint["input_dim"]=20
    checkpoint["output_dim"]=3
    checkpoint["Vref"]=vref
    checkpoint["Ts_ctrl"]=dt
    checkpoint["angle_low"]=low
    checkpoint["angle_high"]=high
    checkpoint["config"]={**checkpoint.get("config",{}),"du_max":[3.,3.,3.],
                          "policy_hidden_dims":dims[1:-1],"post_distill_dpc":True}
    checkpoint["dpc_finetune"]={"best_epoch":best_epoch,"original":str(stu_file),
                                "teacher":str(teacher_dir),"surrogate":str(surrogate_dir),
                                "validation_horizon":validation_horizon,
                                "baseline":baseline,"final":final}
    torch.save(checkpoint,dst/"dynamic_policy_best.pth")
    joblib.dump({"mean":mean_s,"std":std_s,"columns":FEATURE_NAMES},
                dst/"dynamic_policy_input_scaler.pkl")
    for name in ("config.json","runtime_limits.json"):
        p=teacher_dir/name
        if p.exists():shutil.copy2(p,dst/name)
    original_steady=student_dir/"steady_policy_nsga_pref_no_pout_output"
    if original_steady.is_dir():
        steady_copy=outdir/"steady_policy_nsga_pref_no_pout_output"
        steady_copy.mkdir(parents=True,exist_ok=True)
        for name in ("steady_policy_nsga_pref.pth",
                     "steady_policy_input_scaler.pkl","steady_policy_angle_bounds.npy"):
            shutil.copy2(original_steady/name,steady_copy/name)
    pd.DataFrame(history).to_csv(outdir/"dpc_history.csv",index=False,encoding="utf-8-sig")
    report=dict(original_teacher_du_max=[3.,3.,3.],student_structure=dims,
                student_parameters=sum(p.numel() for p in policy.parameters()),
                val_horizon=validation_horizon,Ts_ctrl=dt,best_epoch=best_epoch,
                scenario_count={"train":len(train_np),"validation":len(val_np)},
                baseline=baseline,after_finetune=final,
                note="Plant surrogate-based optimization; check PLECS stability/accuracy before use")
    (outdir/"dpc_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("\n完成。新模型:", (dst/"dynamic_policy_best.pth").resolve())
    print("原始小模型没有被覆盖。注意检查dpc_report.json和PLECS多工况闭环结果。")


if __name__=="__main__":
    main()
