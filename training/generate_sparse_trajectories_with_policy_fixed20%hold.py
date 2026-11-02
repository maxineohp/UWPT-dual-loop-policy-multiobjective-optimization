# -*- coding: utf-8 -*-
"""
generate_dense_angle_trajectories_with_policy.py

生成 PLECS From File 可读取的动态轨迹 CSV。

核心逻辑：
1. 一条轨迹从一个随机初始工况开始；
2. 初始 a,b,c = 该初始工况下稳态最优角附近扰动；
3. 80 ms 后开始动态段；
4. 每 20 ms 进行一次工况阶跃，只改变 Vin 和 Rload，不改变相角；
5. 每 200 us 进行一次相角控制事件；
6. 相角控制事件中：
   - 20%：无相角改变，a,b,c 保持不变；
   - 剩余 80%：发生相角动作；
       * 70%：小步调角，±1°~±3°
       * 20%：中等调角，±3°~±8°
       * 10%：大扰动调角，±8°~±20°
7. 所有事件点相对采样边界延迟 event_delay=1 us。

输出 CSV:
    time,Vin,Rload,a,b,c
"""

import time
import re
import pickle
import random
import joblib
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. 配置区
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # 稳态最优角网络文件夹
    # --------------------------------------------------------
    "policy_dir": "steady_policy_nsga_pref_no_pout_relu_output",
    "policy_pth": "steady_policy_nsga_pref.pth",
    "policy_input_scaler": "steady_policy_input_scaler.pkl",
    "policy_angle_bounds": "steady_policy_angle_bounds.npy",

    # 如果稳态网络输入是 [Vin, Rload, preference]，这里填 preference
    "pref_value": 0.5,

    # auto / direct_angles / normalized01_to_bounds / sigmoid_to_bounds
    "policy_output_mode": "auto",

    # --------------------------------------------------------
    # 轨迹数量与随机性
    # --------------------------------------------------------
    "n_traj": 300,

    # True：每次运行生成完全一样的数据
    # False：每次运行生成不同数据
    "reproducible": True,
    "seed": 42,

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------
    "out_dir": "traj_csv",
    "summary_csv": "trajectory_summary.csv",
    "debug_csv": "trajectory_points_debug.csv",

    # --------------------------------------------------------
    # 时间设置
    # --------------------------------------------------------
    "t_stop": 0.200,
    "t_dynamic_start": 0.080,

    # 输出采样周期 / 相角控制周期
    "Ts_ctrl": 200e-6,

    # 工况阶跃间隔
    "condition_step_interval": 0.020,

    # 事件点相对输出采样边界延迟
    "event_delay": 1e-6,

    # --------------------------------------------------------
    # 工况与角度范围
    # --------------------------------------------------------
    "range": {
        "Vin":   (250.0, 350.0),
        "Rload": (100.0, 150.0),
        "a":     (90.0, 180.0),
        "b":     (90.0, 180.0),
        "c":     (90.0, 150.0),
    },

    # --------------------------------------------------------
    # 初始角度：稳态最优角附近扰动
    # --------------------------------------------------------
    "init_jitter_deg": 3.0,

    # --------------------------------------------------------
    # 每 20 ms 工况阶跃幅度
    # 只改变 Vin 和 Rload，不改变 a,b,c
    # --------------------------------------------------------
    "Vin_step_range": (10.0, 40.0),
    "R_step_range":   (5.0, 20.0),

    # 如果阶跃后碰到边界，使用反向扰动，尽量避免被 clip 后不变化
    "avoid_zero_condition_step": True,

    # --------------------------------------------------------
    # 每 200 us 相角事件设置
    # --------------------------------------------------------

    # 20%：相角完全保持不变
    "hold_angle_prob": 0.20,

    # 剩余 80% 发生相角动作；
    # 在发生动作的样本内部，按下面比例选择动作幅度。
    "angle_step_prob": {
        "small":  0.70,
        "medium": 0.20,
        "large":  0.10,
    },

    "angle_step_range": {
        "small":  (1.0, 3.0),
        "medium": (3.0, 8.0),
        "large":  (8.0, 20.0),
    },

    # 相角动作方向：
    # 以 toward_opt_prob 的概率朝当前工况稳态最优角方向移动；
    # 其余概率随机方向，用于增强动作激励覆盖。
    "toward_opt_prob": 0.70,

    # 是否每次 a/b/c 都同时变化。
    # True：每个非 hold 的 200us 事件中 a,b,c 都变化；
    # False：随机选择 1~3 个角变化。
    "change_all_angles_each_step": True,

    "time_decimals": 6,
    "value_decimals": 6,
}


# ============================================================
# 2. 网络定义与加载
# ============================================================

class MLPPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def extract_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()

    if isinstance(obj, dict):
        for key in ["model_state_dict", "state_dict", "policy_state_dict", "net_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]

        if len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values()):
            return obj

        for _, v in obj.items():
            if isinstance(v, dict) and len(v) > 0 and all(torch.is_tensor(x) for x in v.values()):
                return v

    raise RuntimeError("无法从 .pth 文件中解析 state_dict。")


def infer_mlp_dims_from_state_dict(state_dict):
    weight_items = []
    for k, v in state_dict.items():
        if torch.is_tensor(v) and k.endswith("weight") and v.ndim == 2:
            weight_items.append((k, v))

    if len(weight_items) < 2:
        raise RuntimeError("state_dict 中没有足够的 Linear weight。")

    weight_items = sorted(weight_items, key=lambda kv: natural_key(kv[0]))
    shapes = [tuple(v.shape) for _, v in weight_items]

    input_dim = shapes[0][1]
    hidden_dims = [s[0] for s in shapes[:-1]]
    output_dim = shapes[-1][0]

    return input_dim, hidden_dims, output_dim, weight_items


def remap_state_dict_to_mlp(state_dict, weight_items):
    new_sd = OrderedDict()

    for i, (w_key, w_val) in enumerate(weight_items):
        linear_pos = 2 * i
        b_key = w_key[:-len("weight")] + "bias"

        if b_key not in state_dict:
            raise RuntimeError(f"找不到 bias 参数：{b_key}")

        new_sd[f"net.{linear_pos}.weight"] = w_val
        new_sd[f"net.{linear_pos}.bias"] = state_dict[b_key]

    return new_sd


def load_angle_bounds(policy_dir, cfg):
    path = policy_dir / cfg["policy_angle_bounds"]

    if not path.exists():
        lows = np.array(
            [cfg["range"]["a"][0], cfg["range"]["b"][0], cfg["range"]["c"][0]],
            dtype=np.float32
        )
        highs = np.array(
            [cfg["range"]["a"][1], cfg["range"]["b"][1], cfg["range"]["c"][1]],
            dtype=np.float32
        )
        return lows, highs

    arr = np.load(path).astype(np.float32)

    if arr.shape == (3, 2):
        lows = arr[:, 0]
        highs = arr[:, 1]
    elif arr.shape == (2, 3):
        lows = arr[0, :]
        highs = arr[1, :]
    else:
        raise RuntimeError(f"无法识别 steady_policy_angle_bounds.npy 形状：{arr.shape}")

    return lows.astype(np.float32), highs.astype(np.float32)


def load_scaler_file(scaler_path):
    scaler_path = Path(scaler_path)

    try:
        return joblib.load(scaler_path)
    except Exception as e_joblib:
        try:
            with open(scaler_path, "rb") as f:
                return pickle.load(f)
        except Exception as e_pickle:
            raise RuntimeError(
                "输入 scaler 读取失败。\n"
                f"scaler_path = {scaler_path.resolve()}\n"
                f"joblib.load 报错: {repr(e_joblib)}\n"
                f"pickle.load 报错: {repr(e_pickle)}"
            )


def transform_policy_input(scaler, X):
    X = np.asarray(X, dtype=np.float32)

    if hasattr(scaler, "transform") and callable(scaler.transform):
        return scaler.transform(X).astype(np.float32)

    if hasattr(scaler, "transform_np") and callable(scaler.transform_np):
        return scaler.transform_np(X).astype(np.float32)

    if isinstance(scaler, dict):
        if "mean" not in scaler or "std" not in scaler:
            raise RuntimeError(
                "scaler 是 dict，但缺少 mean/std 字段。\n"
                f"当前字段: {list(scaler.keys())}"
            )

        mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, -1)
        std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, -1)
        std = np.where(np.abs(std) < 1e-12, 1.0, std)

        if mean.shape[1] != X.shape[1]:
            raise RuntimeError(
                "scaler 的 mean/std 维度与当前策略网络输入维度不一致。\n"
                f"X.shape = {X.shape}, mean.shape = {mean.shape}, std.shape = {std.shape}\n"
                f"scaler columns = {scaler.get('columns', None)}"
            )

        return ((X - mean) / std).astype(np.float32)

    raise RuntimeError(
        "无法识别输入 scaler 类型。\n"
        f"type(scaler) = {type(scaler)}"
    )


def load_torch_checkpoint(pth_path):
    pth_path = Path(pth_path)

    try:
        return torch.load(pth_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(pth_path, map_location="cpu")
    except Exception as e1:
        try:
            return torch.load(pth_path, map_location="cpu")
        except Exception as e2:
            raise RuntimeError(
                "策略网络 .pth 文件读取失败。\n"
                f"pth_path = {pth_path.resolve()}\n"
                f"torch.load(weights_only=False) 报错: {repr(e1)}\n"
                f"torch.load(旧写法) 报错: {repr(e2)}"
            )


def load_policy(cfg):
    policy_dir = Path(cfg["policy_dir"])
    pth_path = policy_dir / cfg["policy_pth"]
    scaler_path = policy_dir / cfg["policy_input_scaler"]

    if not pth_path.exists():
        raise FileNotFoundError(f"找不到策略网络权重：{pth_path.resolve()}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"找不到输入 scaler：{scaler_path.resolve()}")

    scaler = load_scaler_file(scaler_path)

    obj = load_torch_checkpoint(pth_path)
    sd = extract_state_dict(obj)

    clean_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_sd[k] = v

    input_dim, hidden_dims, output_dim, weight_items = infer_mlp_dims_from_state_dict(clean_sd)

    model = MLPPolicy(input_dim, hidden_dims, output_dim)
    model.load_state_dict(remap_state_dict_to_mlp(clean_sd, weight_items), strict=True)
    model.eval()

    angle_low, angle_high = load_angle_bounds(policy_dir, cfg)

    print("稳态最优角网络加载完成：")
    print(f"  policy_dir  = {policy_dir.resolve()}")
    print(f"  input_dim   = {input_dim}")
    print(f"  hidden_dims = {hidden_dims}")
    print(f"  output_dim  = {output_dim}")
    print(f"  angle_low   = {angle_low}")
    print(f"  angle_high  = {angle_high}")

    return model, scaler, input_dim, angle_low, angle_high


def make_policy_input(Vin_array, R_array, input_dim, cfg):
    Vin_array = np.asarray(Vin_array, dtype=np.float32).reshape(-1)
    R_array = np.asarray(R_array, dtype=np.float32).reshape(-1)

    if input_dim == 2:
        X = np.column_stack([Vin_array, R_array])
    elif input_dim == 3:
        pref = np.full(len(Vin_array), float(cfg["pref_value"]), dtype=np.float32)
        X = np.column_stack([Vin_array, R_array, pref])
    else:
        raise RuntimeError(f"当前 input_dim={input_dim}，代码只支持 2维或3维输入。")

    return X.astype(np.float32)


def map_policy_output(raw, angle_low, angle_high, mode):
    raw = np.asarray(raw, dtype=np.float32)
    raw3 = raw[:, :3]

    if mode == "direct_angles":
        angles = raw3

    elif mode == "normalized01_to_bounds":
        y = np.clip(raw3, 0.0, 1.0)
        angles = angle_low + y * (angle_high - angle_low)

    elif mode == "sigmoid_to_bounds":
        y = 1.0 / (1.0 + np.exp(-raw3))
        angles = angle_low + y * (angle_high - angle_low)

    elif mode == "auto":
        in_angle_range = np.mean((raw3 >= (angle_low - 5.0)) & (raw3 <= (angle_high + 5.0)))
        in_01_range = np.mean((raw3 >= -0.05) & (raw3 <= 1.05))

        if in_angle_range > 0.95 and np.mean(raw3) > 10.0:
            angles = raw3
        elif in_01_range > 0.95:
            y = np.clip(raw3, 0.0, 1.0)
            angles = angle_low + y * (angle_high - angle_low)
        else:
            y = 1.0 / (1.0 + np.exp(-raw3))
            angles = angle_low + y * (angle_high - angle_low)

    else:
        raise ValueError(f"Unknown policy_output_mode: {mode}")

    return angles.astype(np.float32)


@torch.no_grad()
def predict_opt_angles(model, scaler, input_dim, angle_low, angle_high, Vin, Rload, cfg):
    X = make_policy_input([Vin], [Rload], input_dim, cfg)
    Xs = transform_policy_input(scaler, X)
    raw = model(torch.from_numpy(Xs)).cpu().numpy()
    angles = map_policy_output(raw, angle_low, angle_high, cfg["policy_output_mode"])[0]

    return float(angles[0]), float(angles[1]), float(angles[2])


# ============================================================
# 3. 工具函数
# ============================================================

def set_seed(cfg):
    if cfg["reproducible"]:
        seed = int(cfg["seed"])
    else:
        seed = int(time.time() * 1000) % (2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed


def clip_value(x, lo, hi):
    return max(float(lo), min(float(hi), float(x)))


def clip_all(Vin, Rload, a, b, c, cfg):
    Vin = clip_value(Vin, *cfg["range"]["Vin"])
    Rload = clip_value(Rload, *cfg["range"]["Rload"])
    a = clip_value(a, *cfg["range"]["a"])
    b = clip_value(b, *cfg["range"]["b"])
    c = clip_value(c, *cfg["range"]["c"])
    return Vin, Rload, a, b, c


def uniform_range(name, cfg):
    lo, hi = cfg["range"][name]
    return random.uniform(float(lo), float(hi))


def signed_delta(delta_range):
    lo, hi = delta_range
    mag = random.uniform(float(lo), float(hi))
    return random.choice([-1.0, 1.0]) * mag


def jitter_angle(opt, jitter):
    return opt + random.uniform(-float(jitter), float(jitter))


def bounded_add_delta(x, delta, lo, hi):
    y = x + delta

    if y < lo or y > hi:
        y = x - delta

    y = clip_value(y, lo, hi)
    return y


def choose_angle_step_class(cfg):
    probs = cfg["angle_step_prob"]
    names = list(probs.keys())
    weights = np.array([float(probs[n]) for n in names], dtype=float)
    weights = weights / weights.sum()
    return random.choices(names, weights=weights, k=1)[0]


def random_mag(range_pair):
    lo, hi = range_pair
    return random.uniform(float(lo), float(hi))


def one_angle_action(current, target, lo, hi, step_class, cfg):
    mag = random_mag(cfg["angle_step_range"][step_class])

    if random.random() < float(cfg["toward_opt_prob"]):
        err = target - current
        if abs(err) < 1e-9:
            sgn = random.choice([-1.0, 1.0])
        else:
            sgn = np.sign(err)
    else:
        sgn = random.choice([-1.0, 1.0])

    delta = sgn * mag
    new_val = bounded_add_delta(current, delta, lo, hi)

    return new_val


def apply_angle_control_step(a, b, c, a_opt, b_opt, c_opt, cfg):
    """
    每 200 us 进入一次相角控制事件。

    先以 hold_angle_prob 的概率保持相角不变；
    否则再按 small/medium/large 比例产生相角动作。
    """
    if random.random() < float(cfg["hold_angle_prob"]):
        return a, b, c, "hold", "none"

    step_class = choose_angle_step_class(cfg)

    a_lo, a_hi = cfg["range"]["a"]
    b_lo, b_hi = cfg["range"]["b"]
    c_lo, c_hi = cfg["range"]["c"]

    if cfg["change_all_angles_each_step"]:
        a = one_angle_action(a, a_opt, a_lo, a_hi, step_class, cfg)
        b = one_angle_action(b, b_opt, b_lo, b_hi, step_class, cfg)
        c = one_angle_action(c, c_opt, c_lo, c_hi, step_class, cfg)
        changed = "abc"
    else:
        angle_names = ["a", "b", "c"]
        n_change = random.randint(1, 3)
        selected = set(random.sample(angle_names, n_change))

        if "a" in selected:
            a = one_angle_action(a, a_opt, a_lo, a_hi, step_class, cfg)
        if "b" in selected:
            b = one_angle_action(b, b_opt, b_lo, b_hi, step_class, cfg)
        if "c" in selected:
            c = one_angle_action(c, c_opt, c_lo, c_hi, step_class, cfg)

        changed = "".join([x for x in angle_names if x in selected])

    return a, b, c, step_class, changed


def apply_condition_step(Vin, Rload, cfg):
    """
    每 20 ms 工况阶跃一次。
    只改变 Vin 和 Rload，不改变 a,b,c。
    """
    Vin_lo, Vin_hi = cfg["range"]["Vin"]
    R_lo, R_hi = cfg["range"]["Rload"]

    dVin = signed_delta(cfg["Vin_step_range"])
    dR = signed_delta(cfg["R_step_range"])

    if cfg["avoid_zero_condition_step"]:
        Vin_new = bounded_add_delta(Vin, dVin, Vin_lo, Vin_hi)
        R_new = bounded_add_delta(Rload, dR, R_lo, R_hi)
    else:
        Vin_new = clip_value(Vin + dVin, Vin_lo, Vin_hi)
        R_new = clip_value(Rload + dR, R_lo, R_hi)

    return Vin_new, R_new


def build_row(t, Vin, Rload, a, b, c, event_type, opt_angles, cfg, step_class="", changed=""):
    a_opt, b_opt, c_opt = opt_angles
    Vin, Rload, a, b, c = clip_all(Vin, Rload, a, b, c, cfg)

    return {
        "time": round(float(t), int(cfg["time_decimals"])),
        "Vin": round(float(Vin), int(cfg["value_decimals"])),
        "Rload": round(float(Rload), int(cfg["value_decimals"])),
        "a": round(float(a), int(cfg["value_decimals"])),
        "b": round(float(b), int(cfg["value_decimals"])),
        "c": round(float(c), int(cfg["value_decimals"])),
        "event_type": event_type,
        "angle_step_class": step_class,
        "changed_angles": changed,
        "a_opt": round(float(a_opt), int(cfg["value_decimals"])),
        "b_opt": round(float(b_opt), int(cfg["value_decimals"])),
        "c_opt": round(float(c_opt), int(cfg["value_decimals"])),
    }


def is_condition_boundary(ctrl_index, cfg):
    n = int(round(float(cfg["condition_step_interval"]) / float(cfg["Ts_ctrl"])))
    if n <= 0:
        raise RuntimeError("condition_step_interval / Ts_ctrl 配置错误。")
    return (ctrl_index % n) == 0


# ============================================================
# 4. 单条轨迹生成
# ============================================================

def generate_one_trajectory(traj_id, model, scaler, input_dim, angle_low, angle_high, cfg):
    rows = []

    t_stop = float(cfg["t_stop"])
    t_start = float(cfg["t_dynamic_start"])
    Ts_ctrl = float(cfg["Ts_ctrl"])
    event_delay = float(cfg["event_delay"])

    # --------------------------------------------------------
    # 0.000000 s：初始随机工况，角度在稳态最优角附近
    # --------------------------------------------------------
    Vin = uniform_range("Vin", cfg)
    Rload = uniform_range("Rload", cfg)

    a_opt, b_opt, c_opt = predict_opt_angles(
        model, scaler, input_dim, angle_low, angle_high, Vin, Rload, cfg
    )

    a = jitter_angle(a_opt, cfg["init_jitter_deg"])
    b = jitter_angle(b_opt, cfg["init_jitter_deg"])
    c = jitter_angle(c_opt, cfg["init_jitter_deg"])

    Vin, Rload, a, b, c = clip_all(Vin, Rload, a, b, c, cfg)

    rows.append(build_row(
        0.0, Vin, Rload, a, b, c,
        "initial",
        (a_opt, b_opt, c_opt),
        cfg,
    ))

    # --------------------------------------------------------
    # 80 ms 到 200 ms：
    # 每 200 us 一个事件点。
    # 若该点是 20 ms 工况边界：只改变 Vin/Rload；
    # 否则：进入相角控制事件，其中 20% 概率保持相角不变。
    # --------------------------------------------------------
    n_ctrl = int(round((t_stop - t_start) / Ts_ctrl))

    for k in range(n_ctrl):
        sample_boundary = t_start + k * Ts_ctrl
        event_time = sample_boundary + event_delay

        if event_time >= t_stop - 1e-12:
            break

        if is_condition_boundary(k, cfg):
            # 每 20 ms 工况阶跃：只变 Vin/R，不改变相角
            Vin, Rload = apply_condition_step(Vin, Rload, cfg)

            # 新工况下更新目标最优角，但当前 a,b,c 不变
            a_opt, b_opt, c_opt = predict_opt_angles(
                model, scaler, input_dim, angle_low, angle_high, Vin, Rload, cfg
            )

            event_type = "condition_step_Vin_R_only"
            step_class = ""
            changed = "Vin_R"

        else:
            a_old, b_old, c_old = a, b, c

            a, b, c, step_class, changed = apply_angle_control_step(
                a, b, c, a_opt, b_opt, c_opt, cfg
            )

            Vin, Rload, a, b, c = clip_all(Vin, Rload, a, b, c, cfg)

            if step_class == "hold":
                event_type = "angle_hold_step"
            else:
                event_type = "angle_control_step"

            # 防止由于边界 clip 导致理论动作后实际没变；
            # 如果实际完全没变，也按 hold 记录。
            if abs(a - a_old) + abs(b - b_old) + abs(c - c_old) < 1e-12:
                event_type = "angle_hold_step"
                step_class = "hold"
                changed = "none"

        rows.append(build_row(
            event_time, Vin, Rload, a, b, c,
            event_type,
            (a_opt, b_opt, c_opt),
            cfg,
            step_class=step_class,
            changed=changed,
        ))

    # --------------------------------------------------------
    # 0.200000 s：重复最后工况
    # --------------------------------------------------------
    rows.append(build_row(
        t_stop, Vin, Rload, a, b, c,
        "hold_last",
        (a_opt, b_opt, c_opt),
        cfg,
    ))

    df_full = pd.DataFrame(rows)
    df_csv = df_full[["time", "Vin", "Rload", "a", "b", "c"]].copy()

    return df_full, df_csv


# ============================================================
# 5. 主程序
# ============================================================

def main():
    cfg = CONFIG
    used_seed = set_seed(cfg)

    model, scaler, input_dim, angle_low, angle_high = load_policy(cfg)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    debug_all = []

    for traj_id in range(1, int(cfg["n_traj"]) + 1):
        df_full, df_csv = generate_one_trajectory(
            traj_id,
            model, scaler, input_dim, angle_low, angle_high,
            cfg,
        )

        filename = f"traj_{traj_id:04d}.csv"
        path = out_dir / filename

        df_csv.to_csv(path, index=False, encoding="utf-8")

        df_debug = df_full.copy()
        df_debug.insert(0, "traj_id", traj_id)
        debug_all.append(df_debug)

        stat_df = df_full[df_full["event_type"] != "hold_last"].copy()
        event_counts = stat_df["event_type"].value_counts().to_dict()
        class_counts = stat_df["angle_step_class"].value_counts().to_dict()

        # 统计有效非零相角动作
        da = df_csv["a"].diff().fillna(0.0).abs()
        db = df_csv["b"].diff().fillna(0.0).abs()
        dc = df_csv["c"].diff().fillna(0.0).abs()
        dsum = da + db + dc

        dVin = df_csv["Vin"].diff().fillna(0.0).abs()
        dR = df_csv["Rload"].diff().fillna(0.0).abs()
        dcond = dVin + dR

        summaries.append({
            "traj_id": traj_id,
            "filename": str(path),
            "n_rows": len(df_csv),
            "t_start": float(df_csv["time"].iloc[0]),
            "t_stop": float(df_csv["time"].iloc[-1]),

            "Vin_min": float(stat_df["Vin"].min()),
            "Vin_max": float(stat_df["Vin"].max()),
            "Rload_min": float(stat_df["Rload"].min()),
            "Rload_max": float(stat_df["Rload"].max()),

            "a_min": float(stat_df["a"].min()),
            "a_max": float(stat_df["a"].max()),
            "b_min": float(stat_df["b"].min()),
            "b_max": float(stat_df["b"].max()),
            "c_min": float(stat_df["c"].min()),
            "c_max": float(stat_df["c"].max()),

            "count_initial": int(event_counts.get("initial", 0)),
            "count_condition_step": int(event_counts.get("condition_step_Vin_R_only", 0)),
            "count_angle_control_step": int(event_counts.get("angle_control_step", 0)),
            "count_angle_hold_step": int(event_counts.get("angle_hold_step", 0)),
            "count_hold_last": int(event_counts.get("hold_last", 0)),

            "count_angle_small": int(class_counts.get("small", 0)),
            "count_angle_medium": int(class_counts.get("medium", 0)),
            "count_angle_large": int(class_counts.get("large", 0)),
            "count_angle_hold": int(class_counts.get("hold", 0)),

            "nonzero_angle_action_count": int((dsum > 1e-9).sum()),
            "zero_angle_action_count": int((dsum <= 1e-9).sum()),
            "nonzero_condition_count": int((dcond > 1e-9).sum()),
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(cfg["summary_csv"], index=False, encoding="utf-8")

    debug_df = pd.concat(debug_all, ignore_index=True)
    debug_df.to_csv(cfg["debug_csv"], index=False, encoding="utf-8")

    print("轨迹生成完成。")
    print(f"输出文件夹: {out_dir.resolve()}")
    print(f"总轨迹数: {cfg['n_traj']}")
    print(f"reproducible: {cfg['reproducible']}")
    print(f"本次 seed: {used_seed}")
    print(f"summary_csv: {Path(cfg['summary_csv']).resolve()}")
    print(f"debug_csv: {Path(cfg['debug_csv']).resolve()}")

    print("\n轨迹设置：")
    print(f"动态开始时间: {cfg['t_dynamic_start']} s")
    print(f"仿真结束时间: {cfg['t_stop']} s")
    print(f"相角控制周期 Ts_ctrl: {cfg['Ts_ctrl']} s")
    print(f"工况阶跃间隔: {cfg['condition_step_interval']} s")
    print(f"事件延迟: {cfg['event_delay']} s")
    print(f"无相角改变概率 hold_angle_prob: {cfg['hold_angle_prob']}")

    print("\n非 hold 相角动作比例设置：")
    for k, v in cfg["angle_step_prob"].items():
        print(f"{k}: {v:.2f}")

    print("\n前 3 条轨迹统计：")
    print(summary_df.head(3))

    total_angle_actions = int(summary_df["count_angle_control_step"].sum())
    total_angle_hold = int(summary_df["count_angle_hold_step"].sum())
    total_condition_steps = int(summary_df["count_condition_step"].sum())

    total_angle_events = total_angle_actions + total_angle_hold

    print("\n总体统计：")
    print(f"总相角动作事件数: {total_angle_actions}")
    print(f"总相角保持事件数: {total_angle_hold}")
    print(f"总相角事件数: {total_angle_events}")
    print(f"总工况阶跃事件数: {total_condition_steps}")

    if total_angle_events > 0:
        print(f"相角保持比例: {total_angle_hold / total_angle_events:.2%}")
        print(f"相角动作比例: {total_angle_actions / total_angle_events:.2%}")

    print(f"平均每条轨迹相角动作事件数: {total_angle_actions / cfg['n_traj']:.1f}")
    print(f"平均每条轨迹相角保持事件数: {total_angle_hold / cfg['n_traj']:.1f}")
    print(f"平均每条轨迹工况阶跃事件数: {total_condition_steps / cfg['n_traj']:.1f}")


if __name__ == "__main__":
    main()