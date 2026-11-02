# -*- coding: utf-8 -*-
"""
export_plecs_headers.py

作用：
    将 PyTorch 训练得到的稳态策略网络和动态策略网络导出为 PLECS C-Script 可用的 .h 头文件。

输出：
    plecs_headers/steady_policy_weights.h
    plecs_headers/dynamic_policy_weights.h

核心修正：
    先读取 scaler，确定网络真实输入维度；
    再从 .pth 文件中筛选第一层输入维度匹配、最后一层输出维度为 3 的 MLP 权重链。
    这样可以避免 checkpoint 中存在多个网络时误选到输入维度为 75 的网络。

默认目录结构：
    当前目录/
    ├─ steady_policy_nsga_pref_no_pout_relu_output/
    │  ├─ steady_policy_nsga_pref.pth
    │  ├─ steady_policy_input_scaler.pkl
    │  └─ steady_policy_angle_bounds.npy
    ├─ dynamic_policy_output/
    │  ├─ dynamic_policy_best.pth
    │  ├─ dynamic_policy_input_scaler.pkl
    │  ├─ config.json
    │  └─ runtime_limits.json
    └─ export_plecs_headers.py

运行：
    python export_plecs_headers.py
"""

import json
import pickle
import re
from pathlib import Path
from collections import OrderedDict

import joblib
import numpy as np
import torch


# ============================================================
# 1. 用户配置区
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # 稳态最优角网络
    # --------------------------------------------------------
    "steady_dir": "lightweight_models_dpc/steady_policy_nsga_pref_no_pout_output",
    "steady_pth": "steady_policy_nsga_pref.pth",
    "steady_input_scaler": "steady_policy_input_scaler.pkl",
    "steady_angle_bounds": "steady_policy_angle_bounds.npy",

    # 如果稳态网络输入是 [Vin, Rload, preference]，这里填 preference
    "steady_pref_value": 1,

    # 稳态网络输出模式：
    # auto / direct_angles / normalized01_to_bounds / sigmoid_to_bounds
    #
    # 如果 auto 判断错了，手动改：
    #   direct_angles              网络直接输出 a,b,c 角度
    #   normalized01_to_bounds     网络输出 0~1，再映射到角度范围
    #   sigmoid_to_bounds          网络输出任意实数，先 sigmoid，再映射到角度范围
    "steady_output_mode": "auto",

    # --------------------------------------------------------
    # 动态策略网络
    # --------------------------------------------------------
    "dynamic_dir": "dynamic_policy_small_dpc_output",
    "dynamic_pth": "dynamic_policy_best.pth",
    "dynamic_input_scaler": "dynamic_policy_input_scaler.pkl",
    "dynamic_config_json": "config.json",
    "dynamic_runtime_limits_json": "runtime_limits.json",

    # 动态策略网络输出是否需要 tanh 限幅：
    # 如果训练时 forward 是 du = du_max * tanh(raw)，这里保持 True；
    # 如果训练时网络直接输出 da,db,dc，改成 False。
    "dynamic_output_tanh": False,

    # 如果 config/runtime_limits 中找不到动作限幅，就使用该默认值，单位：deg/control step
    "default_du_max_deg": 3.0,

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------
    "out_dir": "plecs_headers_small",

    # C 数组数字精度
    "float_format": ".10g",

    # 是否打印所有 Linear 层候选形状
    "print_all_linear_shapes": True,
}


# ============================================================
# 2. 基础工具函数
# ============================================================

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def load_torch_checkpoint(path):
    path = Path(path)

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as e1:
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e2:
            raise RuntimeError(
                "torch checkpoint 读取失败。\n"
                f"path = {path.resolve()}\n"
                f"第一次报错: {repr(e1)}\n"
                f"第二次报错: {repr(e2)}"
            )


def is_tensor_dict(d):
    if not isinstance(d, dict) or len(d) == 0:
        return False
    return all(torch.is_tensor(v) for v in d.values())


def collect_state_dict_candidates(obj, prefix="root", max_depth=5):
    """
    从 checkpoint 中递归收集所有可能的 state_dict。

    返回：
        [(name, state_dict), ...]
    """
    candidates = []

    if max_depth < 0:
        return candidates

    if isinstance(obj, torch.nn.Module):
        candidates.append((prefix + ".module_state_dict", obj.state_dict()))
        return candidates

    if isinstance(obj, dict):
        # 常见 key 优先
        preferred_keys = [
            "model_state_dict",
            "state_dict",
            "policy_state_dict",
            "net_state_dict",
            "dynamic_policy_state_dict",
            "steady_policy_state_dict",
            "actor_state_dict",
        ]

        for key in preferred_keys:
            if key in obj and is_tensor_dict(obj[key]):
                candidates.append((prefix + "." + key, obj[key]))

        if is_tensor_dict(obj):
            candidates.append((prefix, obj))

        # 递归搜索子 dict
        for k, v in obj.items():
            if isinstance(v, dict):
                candidates.extend(
                    collect_state_dict_candidates(
                        v,
                        prefix=prefix + "." + str(k),
                        max_depth=max_depth - 1,
                    )
                )

    return candidates


def clean_state_dict(sd):
    """
    把 torch tensor 转成 numpy，并去掉 module. 前缀。
    """
    out = OrderedDict()

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue

        key = str(k)

        if key.startswith("module."):
            key = key[len("module."):]

        out[key] = v.detach().cpu().numpy().astype(np.float64)

    return out


def get_linear_weight_items(sd):
    weight_items = []

    for k, v in sd.items():
        if isinstance(v, np.ndarray) and k.endswith("weight") and v.ndim == 2:
            weight_items.append((k, v))

    weight_items = sorted(weight_items, key=lambda kv: natural_key(kv[0]))
    return weight_items


def extract_mlp_chain_from_state_dict(
    sd,
    expected_input_dim,
    expected_output_dim=3,
    verbose=False,
):
    """
    从一个 numpy state_dict 中提取输入/输出维度匹配的 MLP Linear 层链。

    Linear weight 形状按 PyTorch 规则：
        W.shape = [out_dim, in_dim]

    返回：
        dims, weights, biases, selected_keys
    """
    weight_items = get_linear_weight_items(sd)

    if verbose:
        print("  Linear weight shapes:")
        for k, W in weight_items:
            print(f"    {k}: {tuple(W.shape)}")

    if len(weight_items) == 0:
        return None

    all_chains = []

    for start_idx, (k0, W0) in enumerate(weight_items):
        # 第一层必须匹配 scaler 输入维度
        if W0.shape[1] != expected_input_dim:
            continue

        chain = [(k0, W0)]
        current_dim = W0.shape[0]

        # 贪心向后找维度连续的 Linear 层
        for k, W in weight_items[start_idx + 1:]:
            if W.shape[1] == current_dim:
                chain.append((k, W))
                current_dim = W.shape[0]

                if current_dim == expected_output_dim:
                    # 这里不立即 break 也可以，但 MLP 最后一层通常就是 output_dim
                    break

        # 最后一层输出必须是 expected_output_dim
        if len(chain) >= 1 and chain[-1][1].shape[0] == expected_output_dim:
            all_chains.append(chain)

    if len(all_chains) == 0:
        return None

    # 优先选择层数最多的链
    chain = sorted(all_chains, key=lambda x: len(x), reverse=True)[0]

    weights = []
    biases = []
    selected_keys = []

    for w_key, W in chain:
        b_key = w_key[:-len("weight")] + "bias"

        if b_key not in sd:
            raise RuntimeError(
                f"找到了 weight 但找不到对应 bias。\n"
                f"weight key = {w_key}\n"
                f"bias key   = {b_key}"
            )

        b = sd[b_key]

        if not isinstance(b, np.ndarray) or b.ndim != 1:
            raise RuntimeError(
                f"bias 维度异常。\n"
                f"bias key = {b_key}\n"
                f"shape = {getattr(b, 'shape', None)}"
            )

        weights.append(W.astype(np.float64))
        biases.append(b.astype(np.float64))
        selected_keys.append((w_key, b_key, tuple(W.shape), tuple(b.shape)))

    dims = [weights[0].shape[1]]
    for W in weights:
        dims.append(W.shape[0])

    return dims, weights, biases, selected_keys


def extract_mlp_from_checkpoint(
    ckpt_obj,
    expected_input_dim,
    expected_output_dim=3,
    name="network",
    print_all_shapes=True,
):
    """
    从 checkpoint 中自动筛选输入维度匹配的 MLP。
    """
    candidates = collect_state_dict_candidates(ckpt_obj)

    if len(candidates) == 0:
        raise RuntimeError(f"{name}: checkpoint 中没有找到任何 state_dict 候选。")

    print(f"\n[{name}] state_dict 候选数量: {len(candidates)}")

    matched = []

    for cand_name, sd_torch in candidates:
        sd = clean_state_dict(sd_torch)

        print(f"\n[{name}] 检查候选: {cand_name}")

        result = extract_mlp_chain_from_state_dict(
            sd,
            expected_input_dim=expected_input_dim,
            expected_output_dim=expected_output_dim,
            verbose=print_all_shapes,
        )

        if result is not None:
            dims, weights, biases, selected_keys = result
            matched.append((cand_name, dims, weights, biases, selected_keys))
            print(f"  [MATCH] dims = {dims}")

    if len(matched) == 0:
        # 汇总所有 Linear 形状，方便排查
        msg_lines = []
        msg_lines.append(f"{name}: 没有找到输入/输出维度匹配的 MLP。")
        msg_lines.append(f"expected_input_dim  = {expected_input_dim}")
        msg_lines.append(f"expected_output_dim = {expected_output_dim}")
        msg_lines.append("")
        msg_lines.append("所有候选 state_dict 的 Linear 层形状如下：")

        for cand_name, sd_torch in candidates:
            sd = clean_state_dict(sd_torch)
            weight_items = get_linear_weight_items(sd)

            msg_lines.append(f"\n候选: {cand_name}")
            if len(weight_items) == 0:
                msg_lines.append("  无 Linear weight")
            else:
                for k, W in weight_items:
                    msg_lines.append(f"  {k}: {tuple(W.shape)}")

        msg_lines.append("")
        msg_lines.append("请检查：")
        msg_lines.append("1. .pth 是否是对应网络；")
        msg_lines.append("2. .pkl scaler 是否和 .pth 属于同一次训练；")
        msg_lines.append("3. 稳态网络输入是否确实为 [Vin,Rload] 或 [Vin,Rload,pref]；")
        msg_lines.append("4. 动态策略网络输入维度是否和训练代码一致。")

        raise RuntimeError("\n".join(msg_lines))

    # 多个匹配时，选层数最多的；若层数相同，选第一个
    matched = sorted(matched, key=lambda x: len(x[1]), reverse=True)
    cand_name, dims, weights, biases, selected_keys = matched[0]

    print(f"\n[{name}] 最终选中候选: {cand_name}")
    print(f"[{name}] dims = {dims}")
    print(f"[{name}] 选中的层：")
    for w_key, b_key, w_shape, b_shape in selected_keys:
        print(f"  {w_key}: {w_shape}, {b_key}: {b_shape}")

    return dims, weights, biases


def load_scaler_file(path, expected_dim=None):
    """
    读取 scaler。

    支持：
    1. sklearn StandardScaler：mean_, scale_
    2. dict：mean/std 或 mean_/scale_
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到 scaler 文件：{path.resolve()}")

    try:
        scaler = joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            scaler = pickle.load(f)

    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.asarray(scaler.mean_, dtype=np.float64).reshape(-1)
        std = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)

    elif isinstance(scaler, dict):
        if "mean" in scaler:
            mean = np.asarray(scaler["mean"], dtype=np.float64).reshape(-1)
        elif "mean_" in scaler:
            mean = np.asarray(scaler["mean_"], dtype=np.float64).reshape(-1)
        else:
            raise RuntimeError(
                f"scaler 是 dict，但没有 mean/mean_。\n"
                f"path={path.resolve()}\n"
                f"keys={list(scaler.keys())}"
            )

        if "std" in scaler:
            std = np.asarray(scaler["std"], dtype=np.float64).reshape(-1)
        elif "scale" in scaler:
            std = np.asarray(scaler["scale"], dtype=np.float64).reshape(-1)
        elif "scale_" in scaler:
            std = np.asarray(scaler["scale_"], dtype=np.float64).reshape(-1)
        else:
            raise RuntimeError(
                f"scaler 是 dict，但没有 std/scale/scale_。\n"
                f"path={path.resolve()}\n"
                f"keys={list(scaler.keys())}"
            )
    else:
        raise RuntimeError(
            f"无法识别 scaler 类型。\n"
            f"path={path.resolve()}\n"
            f"type={type(scaler)}"
        )

    std = np.where(np.abs(std) < 1e-12, 1.0, std)

    if expected_dim is not None and len(mean) != int(expected_dim):
        raise RuntimeError(
            "scaler 维度不匹配。\n"
            f"path = {path.resolve()}\n"
            f"scaler_dim = {len(mean)}\n"
            f"expected_dim = {expected_dim}"
        )

    print(f"读取 scaler: {path}")
    print(f"  input_dim = {len(mean)}")
    print(f"  mean = {mean}")
    print(f"  std  = {std}")

    return mean, std


def load_angle_bounds(path):
    path = Path(path)

    if not path.exists():
        print(f"未找到 angle_bounds 文件，使用默认角度范围: {path}")
        low = np.array([90.0, 90.0, 90.0], dtype=np.float64)
        high = np.array([180.0, 180.0, 150.0], dtype=np.float64)
        return low, high

    arr = np.load(path).astype(np.float64)

    if arr.shape == (3, 2):
        low = arr[:, 0]
        high = arr[:, 1]
    elif arr.shape == (2, 3):
        low = arr[0, :]
        high = arr[1, :]
    else:
        raise RuntimeError(
            f"无法识别 angle_bounds 形状。\n"
            f"path = {path.resolve()}\n"
            f"shape = {arr.shape}"
        )

    print(f"读取 angle_bounds: {path}")
    print(f"  low  = {low}")
    print(f"  high = {high}")

    return low, high


def numpy_mlp_forward(x, weights, biases):
    y = np.asarray(x, dtype=np.float64)

    if y.ndim == 1:
        y = y.reshape(1, -1)

    for i, (W, b) in enumerate(zip(weights, biases)):
        y = y @ W.T + b.reshape(1, -1)
        if i < len(weights) - 1:
            y = np.maximum(y, 0.0)

    return y


def infer_steady_output_mode(
    mode,
    dims,
    weights,
    biases,
    mean,
    std,
    angle_low,
    angle_high,
    pref_value,
):
    if mode != "auto":
        return mode

    if dims[0] not in [2, 3]:
        raise RuntimeError(
            "稳态网络输入维度不是 2 或 3，无法 auto 判断输出模式。\n"
            f"dims = {dims}"
        )

    Vin_grid = np.array([250.0, 275.0, 300.0, 325.0, 350.0], dtype=np.float64)
    R_grid = np.array([100.0, 112.5, 125.0, 137.5, 150.0], dtype=np.float64)

    X_list = []

    for Vin in Vin_grid:
        for R in R_grid:
            if dims[0] == 2:
                X_list.append([Vin, R])
            else:
                X_list.append([Vin, R, pref_value])

    X = np.asarray(X_list, dtype=np.float64)
    Xs = (X - mean.reshape(1, -1)) / std.reshape(1, -1)

    raw = numpy_mlp_forward(Xs, weights, biases)[:, :3]

    in_angle_range = np.mean(
        (raw >= (angle_low.reshape(1, -1) - 5.0)) &
        (raw <= (angle_high.reshape(1, -1) + 5.0))
    )

    in_01_range = np.mean((raw >= -0.05) & (raw <= 1.05))

    print("\n稳态网络输出模式 auto 判断：")
    print(f"  raw min = {raw.min():.6g}")
    print(f"  raw max = {raw.max():.6g}")
    print(f"  raw mean = {raw.mean():.6g}")
    print(f"  in_angle_range = {in_angle_range:.3f}")
    print(f"  in_01_range = {in_01_range:.3f}")

    if in_angle_range > 0.95 and np.mean(raw) > 10.0:
        return "direct_angles"
    elif in_01_range > 0.95:
        return "normalized01_to_bounds"
    else:
        return "sigmoid_to_bounds"


def steady_mode_to_int(mode):
    mapping = {
        "direct_angles": 0,
        "normalized01_to_bounds": 1,
        "sigmoid_to_bounds": 2,
    }

    if mode not in mapping:
        raise RuntimeError(
            f"未知 steady_output_mode: {mode}\n"
            f"可选: {list(mapping.keys())}"
        )

    return mapping[mode]


def load_json_if_exists(path):
    path = Path(path)

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_dynamic_du_max(dynamic_dir, ckpt_obj, cfg):
    """
    从 dynamic checkpoint/config/runtime_limits 中尽量读取 du_max。
    找不到则使用 default_du_max_deg。
    """
    dynamic_dir = Path(dynamic_dir)

    candidates = []

    if isinstance(ckpt_obj, dict):
        candidates.append(ckpt_obj)

        for k in ["config", "runtime_limits", "limits"]:
            if isinstance(ckpt_obj.get(k, None), dict):
                candidates.append(ckpt_obj[k])

    config_json = load_json_if_exists(dynamic_dir / cfg["dynamic_config_json"])
    limits_json = load_json_if_exists(dynamic_dir / cfg["dynamic_runtime_limits_json"])

    candidates.append(config_json)
    candidates.append(limits_json)

    keys = [
        "du_max",
        "du_max_deg",
        "delta_u_max",
        "delta_u_max_deg",
        "du_limit",
        "du_limit_deg",
        "max_du_deg",
        "DU_MAX",
    ]

    for d in candidates:
        if not isinstance(d, dict):
            continue

        for k in keys:
            if k in d:
                val = d[k]

                if isinstance(val, (int, float)):
                    return float(val)

                if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
                    return float(max(abs(float(x)) for x in val))

    return float(cfg["default_du_max_deg"])


# ============================================================
# 3. C 代码导出工具
# ============================================================

def c_number(x, fmt):
    x = float(x)

    if not np.isfinite(x):
        raise RuntimeError(f"数组中出现非有限数: {x}")

    return format(x, fmt)


def c_array_1d(name, arr, fmt=".10g"):
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    vals = ", ".join(c_number(x, fmt) for x in arr)
    return f"static const double {name}[{len(arr)}] = {{{vals}}};\n"


def c_array_int(name, arr):
    arr = list(map(int, arr))
    vals = ", ".join(str(x) for x in arr)
    return f"static const int {name}[{len(arr)}] = {{{vals}}};\n"


def c_array_flat_weight(name, W, fmt=".10g"):
    W = np.asarray(W, dtype=np.float64)
    flat = W.reshape(-1)
    vals = ", ".join(c_number(x, fmt) for x in flat)
    return f"static const double {name}[{flat.size}] = {{{vals}}};\n"


def write_mlp_header(
    path,
    prefix,
    dims,
    weights,
    biases,
    x_mean,
    x_std,
    extra_macros=None,
    extra_arrays=None,
    fmt=".10g",
):
    path = Path(path)
    guard = f"{prefix}_POLICY_WEIGHTS_H"

    lines = []

    lines.append(f"#ifndef {guard}\n")
    lines.append(f"#define {guard}\n\n")
    lines.append("#include <math.h>\n\n")

    lines.append(f"#define {prefix}_N_LAYERS {len(weights)}\n")
    lines.append(f"#define {prefix}_INPUT_DIM {dims[0]}\n")
    lines.append(f"#define {prefix}_OUTPUT_DIM {dims[-1]}\n")
    lines.append(f"#define {prefix}_MAX_DIM {max(dims)}\n")

    if extra_macros:
        for k, v in extra_macros.items():
            if isinstance(v, bool):
                v = 1 if v else 0
            lines.append(f"#define {k} {v}\n")

    lines.append("\n")

    lines.append(c_array_int(f"{prefix}_DIMS", dims))
    lines.append(c_array_1d(f"{prefix}_X_MEAN", x_mean, fmt))
    lines.append(c_array_1d(f"{prefix}_X_STD", x_std, fmt))

    if extra_arrays:
        for name, arr in extra_arrays.items():
            lines.append(c_array_1d(name, arr, fmt))

    lines.append("\n")

    for i, (W, b) in enumerate(zip(weights, biases)):
        lines.append(c_array_flat_weight(f"{prefix}_W{i}", W, fmt))
        lines.append(c_array_1d(f"{prefix}_B{i}", b, fmt))

    lines.append("\n")

    w_ptrs = ", ".join([f"{prefix}_W{i}" for i in range(len(weights))])
    b_ptrs = ", ".join([f"{prefix}_B{i}" for i in range(len(weights))])

    lines.append(f"static const double* {prefix}_W[{prefix}_N_LAYERS] = {{{w_ptrs}}};\n")
    lines.append(f"static const double* {prefix}_B[{prefix}_N_LAYERS] = {{{b_ptrs}}};\n\n")

    lines.append(f"#endif /* {guard} */\n")

    path.write_text("".join(lines), encoding="utf-8")


# ============================================================
# 4. 主程序
# ============================================================

def main():
    cfg = CONFIG

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("导出 PLECS C-Script 神经网络头文件")
    print("=" * 80)

    # ========================================================
    # 4.1 稳态网络导出
    # ========================================================
    print("\n" + "=" * 80)
    print("1. 导出稳态最优角网络")
    print("=" * 80)

    steady_dir = Path(cfg["steady_dir"])
    steady_pth = steady_dir / cfg["steady_pth"]
    steady_scaler_path = steady_dir / cfg["steady_input_scaler"]
    steady_angle_bounds_path = steady_dir / cfg["steady_angle_bounds"]

    if not steady_pth.exists():
        raise FileNotFoundError(f"找不到稳态网络 pth: {steady_pth.resolve()}")

    # 先读取 scaler，确定输入维度
    steady_mean, steady_std = load_scaler_file(steady_scaler_path, expected_dim=None)
    steady_input_dim = len(steady_mean)

    if steady_input_dim not in [2, 3]:
        print(
            "\n[警告] 稳态 scaler 输入维度不是 2 或 3。\n"
            f"steady_input_dim = {steady_input_dim}\n"
            "如果你的稳态网络确实不是 [Vin,Rload] 或 [Vin,Rload,pref] 输入，请检查 C-Script 逻辑。"
        )

    # 读取 checkpoint
    steady_ckpt = load_torch_checkpoint(steady_pth)

    # 按 scaler 输入维度筛选 MLP
    steady_dims, steady_weights, steady_biases = extract_mlp_from_checkpoint(
        steady_ckpt,
        expected_input_dim=steady_input_dim,
        expected_output_dim=3,
        name="steady_policy",
        print_all_shapes=bool(cfg["print_all_linear_shapes"]),
    )

    # 再次确认维度一致
    if steady_dims[0] != steady_input_dim:
        raise RuntimeError(
            "稳态网络输入维度和 scaler 不一致。\n"
            f"steady_dims[0] = {steady_dims[0]}\n"
            f"scaler_dim = {steady_input_dim}"
        )

    # 读取角度范围
    angle_low, angle_high = load_angle_bounds(steady_angle_bounds_path)

    # 判断输出模式
    steady_mode = infer_steady_output_mode(
        cfg["steady_output_mode"],
        steady_dims,
        steady_weights,
        steady_biases,
        steady_mean,
        steady_std,
        angle_low,
        angle_high,
        cfg["steady_pref_value"],
    )

    steady_mode_int = steady_mode_to_int(steady_mode)

    print("\n稳态网络最终配置：")
    print(f"  dims = {steady_dims}")
    print(f"  steady_output_mode = {steady_mode}")
    print(f"  steady_output_mode_int = {steady_mode_int}")

    steady_header_path = out_dir / "steady_policy_weights.h"

    write_mlp_header(
        steady_header_path,
        prefix="STEADY",
        dims=steady_dims,
        weights=steady_weights,
        biases=steady_biases,
        x_mean=steady_mean,
        x_std=steady_std,
        extra_macros={
            "STEADY_OUTPUT_MODE": int(steady_mode_int),
            "STEADY_PREF_VALUE": float(cfg["steady_pref_value"]),
        },
        extra_arrays={
            "STEADY_ANGLE_LOW": angle_low,
            "STEADY_ANGLE_HIGH": angle_high,
        },
        fmt=cfg["float_format"],
    )

    print(f"\n[OK] 已生成: {steady_header_path.resolve()}")

    # ========================================================
    # 4.2 动态策略网络导出
    # ========================================================
    print("\n" + "=" * 80)
    print("2. 导出动态策略网络")
    print("=" * 80)

    dynamic_dir = Path(cfg["dynamic_dir"])
    dynamic_pth = dynamic_dir / cfg["dynamic_pth"]
    dynamic_scaler_path = dynamic_dir / cfg["dynamic_input_scaler"]

    if not dynamic_pth.exists():
        raise FileNotFoundError(f"找不到动态策略网络 pth: {dynamic_pth.resolve()}")

    # 先读取动态策略输入 scaler，确定输入维度
    dyn_mean, dyn_std = load_scaler_file(dynamic_scaler_path, expected_dim=None)
    dyn_input_dim = len(dyn_mean)

    # 读取 checkpoint
    dynamic_ckpt = load_torch_checkpoint(dynamic_pth)

    # 按 scaler 输入维度筛选 MLP
    dyn_dims, dyn_weights, dyn_biases = extract_mlp_from_checkpoint(
        dynamic_ckpt,
        expected_input_dim=dyn_input_dim,
        expected_output_dim=3,
        name="dynamic_policy",
        print_all_shapes=bool(cfg["print_all_linear_shapes"]),
    )

    if dyn_dims[0] != dyn_input_dim:
        raise RuntimeError(
            "动态策略网络输入维度和 scaler 不一致。\n"
            f"dyn_dims[0] = {dyn_dims[0]}\n"
            f"scaler_dim = {dyn_input_dim}"
        )

    du_max = find_dynamic_du_max(dynamic_dir, dynamic_ckpt, cfg)

    print("\n动态策略网络最终配置：")
    print(f"  dims = {dyn_dims}")
    print(f"  dynamic_output_tanh = {cfg['dynamic_output_tanh']}")
    print(f"  du_max = {du_max}")

    dynamic_header_path = out_dir / "dynamic_policy_weights.h"

    write_mlp_header(
        dynamic_header_path,
        prefix="DYN",
        dims=dyn_dims,
        weights=dyn_weights,
        biases=dyn_biases,
        x_mean=dyn_mean,
        x_std=dyn_std,
        extra_macros={
            "DYN_OUTPUT_TANH": bool(cfg["dynamic_output_tanh"]),
            "DYN_DU_MAX": float(du_max),
        },
        extra_arrays=None,
        fmt=cfg["float_format"],
    )

    print(f"\n[OK] 已生成: {dynamic_header_path.resolve()}")

    # ========================================================
    # 4.3 结束提示
    # ========================================================
    print("\n" + "=" * 80)
    print("导出完成")
    print("=" * 80)

    print("\n生成文件：")
    print(f"  {steady_header_path.resolve()}")
    print(f"  {dynamic_header_path.resolve()}")

    print("\n下一步：")
    print("1. 将 steady_policy_weights.h 和 dynamic_policy_weights.h 放到 PLECS 模型同目录；")
    print("2. 在 C-Script Code declarations 中 include：")
    print('       #include "steady_policy_weights.h"')
    print('       #include "dynamic_policy_weights.h"')
    print("3. 如果 PLECS 找不到头文件，就把 .h 文件内容直接复制到 C-Script 的 Code declarations 最前面。")

    print("\n如果稳态网络输出角度明显不对：")
    print("  打开本脚本，把 CONFIG['steady_output_mode'] 从 'auto' 改成：")
    print("    direct_angles")
    print("    normalized01_to_bounds")
    print("    sigmoid_to_bounds")
    print("  三者之一，然后重新导出。")


if __name__ == "__main__":
    main()