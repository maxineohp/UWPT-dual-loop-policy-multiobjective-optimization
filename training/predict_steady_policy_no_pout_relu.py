# -*- coding: utf-8 -*-
"""
predict_steady_policy_no_pout_relu.py

ReLU 稳态优化网络 + no_pout 双代理模型预测脚本。

输入:  Vin, Rload, pref_eff
输出:  a, b, c
代理模型输出: efficiency, Vout, Irms1, Irms2
后处理: Pout_calc = Vout^2 / Rload, Irms_obj = max(Irms1, Irms2)

默认目录:
    steady_policy_nsga_pref_no_pout_output/
        steady_policy_nsga_pref.pth
        steady_policy_input_scaler.pkl
        steady_policy_angle_bounds.npy

    model_output_dual_no_pout/
        model_eff.pth
        model_aux.pth
        x_scaler_shared.pkl
        y_scaler_eff.pkl
        y_scaler_aux.pkl

运行:
    python predict_steady_policy_no_pout_relu.py

输入示例:
    300 128.571 0.5
"""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn


# =========================================================
# 0. 路径配置
# =========================================================
POLICY_DIR = "steady_policy_nsga_pref_no_pout_output1"
SURROGATE_DIR = "model_output_dual_no_pout"

POLICY_MODEL_NAME = "steady_policy_nsga_pref.pth"
POLICY_SCALER_NAME = "steady_policy_input_scaler.pkl"
POLICY_BOUNDS_NAME = "steady_policy_angle_bounds.npy"

EFF_MODEL_NAME = "model_eff.pth"
AUX_MODEL_NAME = "model_aux.pth"
X_SCALER_NAME = "x_scaler_shared.pkl"
Y_SCALER_EFF_NAME = "y_scaler_eff.pkl"
Y_SCALER_AUX_NAME = "y_scaler_aux.pkl"


# =========================================================
# 1. 通用函数
# =========================================================
def safe_torch_load(path, map_location="cpu", weights_only=None):
    """兼容 PyTorch 2.6 的 weights_only 默认行为。"""
    if weights_only is None:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except Exception:
        # 对自己训练保存的 checkpoint，可退回 weights_only=False。
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def check_files(paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")


def get_state_dict(obj):
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


# =========================================================
# 2. ReLU 稳态策略网络
#    必须与 ReLU 训练脚本保持一致
# =========================================================
class SteadyAnglePolicyReLU(nn.Module):
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


# =========================================================
# 3. 代理模型结构
#    注意：代理模型仍是 GELU，因为代理模型是按 GELU 训练的。
# =========================================================
class MLPRegressorGELU(nn.Module):
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
# 4. 稳态网络预测器
# =========================================================
class SteadyPolicyReLUPredictor:
    def __init__(self, policy_dir=POLICY_DIR, use_cuda_if_available=True):
        self.policy_dir = policy_dir
        self.model_path = os.path.join(policy_dir, POLICY_MODEL_NAME)
        self.scaler_path = os.path.join(policy_dir, POLICY_SCALER_NAME)
        self.bounds_path = os.path.join(policy_dir, POLICY_BOUNDS_NAME)

        check_files([self.model_path, self.scaler_path, self.bounds_path])

        checkpoint = safe_torch_load(self.model_path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        use_float64 = bool(cfg.get("use_float64", True))

        if use_float64:
            torch.set_default_dtype(torch.float64)
            self.torch_dtype = torch.float64
            self.np_dtype = np.float64
        else:
            torch.set_default_dtype(torch.float32)
            self.torch_dtype = torch.float32
            self.np_dtype = np.float32

        if use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        scaler_data = joblib.load(self.scaler_path)
        if isinstance(scaler_data, dict):
            self.mean = np.asarray(scaler_data["mean"], dtype=self.np_dtype)
            self.std = np.asarray(scaler_data["std"], dtype=self.np_dtype)
            self.input_columns = scaler_data.get("columns", ["Vin", "Rload", "pref_eff"])
        else:
            self.mean = np.asarray(scaler_data.mean_, dtype=self.np_dtype)
            self.std = np.asarray(scaler_data.scale_, dtype=self.np_dtype)
            self.input_columns = checkpoint.get("input_columns", ["Vin", "Rload", "pref_eff"])
        self.std[self.std < 1e-12] = 1.0

        if "angle_bounds" in checkpoint:
            self.angle_bounds = np.asarray(checkpoint["angle_bounds"], dtype=self.np_dtype)
        else:
            self.angle_bounds = np.load(self.bounds_path).astype(self.np_dtype)

        input_dim = int(checkpoint.get("input_dim", 3))
        hidden_dims = checkpoint.get("hidden_dims", [128, 128, 64, 32])
        dropout = float(checkpoint.get("dropout", 0.0))

        self.policy = SteadyAnglePolicyReLU(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            angle_bounds=self.angle_bounds,
            dropout=dropout,
        ).to(self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()

        print("ReLU 稳态优化网络加载完成")
        print("Policy dir:", self.policy_dir)
        print("Policy hidden dims:", hidden_dims)
        print("Policy device:", self.device)
        print("Policy dtype:", self.torch_dtype)
        print("Policy input columns:", self.input_columns)
        print("Angle bounds:")
        print(f"  a: [{self.angle_bounds[0, 0]}, {self.angle_bounds[0, 1]}]")
        print(f"  b: [{self.angle_bounds[1, 0]}, {self.angle_bounds[1, 1]}]")
        print(f"  c: [{self.angle_bounds[2, 0]}, {self.angle_bounds[2, 1]}]")

    def _scale_input(self, x_raw):
        return (x_raw - self.mean) / self.std

    def predict_one(self, Vin, Rload, pref_eff=0.5):
        pref_eff = float(np.clip(pref_eff, 0.0, 1.0))
        x_raw = np.array([[Vin, Rload, pref_eff]], dtype=self.np_dtype)
        x_scaled = self._scale_input(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=self.torch_dtype, device=self.device)

        with torch.no_grad():
            angle = self.policy(x_tensor).detach().cpu().numpy()[0]

        return {
            "Vin": float(Vin),
            "Rload": float(Rload),
            "pref_eff": float(pref_eff),
            "a": float(angle[0]),
            "b": float(angle[1]),
            "c": float(angle[2]),
        }


# =========================================================
# 5. no_pout 代理模型预测器
# =========================================================
class NoPoutDualSurrogatePredictor:
    def __init__(self, surrogate_dir=SURROGATE_DIR, use_cuda_if_available=True, use_float64=True):
        self.surrogate_dir = surrogate_dir
        self.model_eff_path = os.path.join(surrogate_dir, EFF_MODEL_NAME)
        self.model_aux_path = os.path.join(surrogate_dir, AUX_MODEL_NAME)
        self.x_scaler_path = os.path.join(surrogate_dir, X_SCALER_NAME)
        self.y_scaler_eff_path = os.path.join(surrogate_dir, Y_SCALER_EFF_NAME)
        self.y_scaler_aux_path = os.path.join(surrogate_dir, Y_SCALER_AUX_NAME)

        check_files([
            self.model_eff_path,
            self.model_aux_path,
            self.x_scaler_path,
            self.y_scaler_eff_path,
            self.y_scaler_aux_path,
        ])

        if use_float64:
            torch.set_default_dtype(torch.float64)
            self.torch_dtype = torch.float64
            self.np_dtype = np.float64
        else:
            torch.set_default_dtype(torch.float32)
            self.torch_dtype = torch.float32
            self.np_dtype = np.float32

        if use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.x_scaler = joblib.load(self.x_scaler_path)
        self.y_scaler_eff = joblib.load(self.y_scaler_eff_path)
        self.y_scaler_aux = joblib.load(self.y_scaler_aux_path)

        self.model_eff = MLPRegressorGELU(5, 1, hidden_dims=(256, 256, 128, 64), dropout=0.0).to(self.device)
        self.model_aux = MLPRegressorGELU(5, 3, hidden_dims=(256, 256, 128, 64), dropout=0.0).to(self.device)

        eff_obj = safe_torch_load(self.model_eff_path, map_location=self.device, weights_only=True)
        aux_obj = safe_torch_load(self.model_aux_path, map_location=self.device, weights_only=True)

        self.model_eff.load_state_dict(get_state_dict(eff_obj))
        self.model_aux.load_state_dict(get_state_dict(aux_obj))
        self.model_eff.eval()
        self.model_aux.eval()

        print("\nno_pout 双代理模型加载完成")
        print("Surrogate dir:", self.surrogate_dir)
        print("Surrogate device:", self.device)
        print("Surrogate dtype:", self.torch_dtype)
        print("Surrogate input:  [Vin, Rload, a, b, c]")
        print("Surrogate output: efficiency, Vout, Irms1, Irms2")
        print("Pout_calc = Vout^2 / Rload")

    def predict_batch(self, X_raw):
        X_raw = np.asarray(X_raw, dtype=self.np_dtype)
        if X_raw.ndim != 2 or X_raw.shape[1] != 5:
            raise ValueError("X_raw 必须是 [N,5]，每行 [Vin, Rload, a, b, c]。")

        X_scaled = self.x_scaler.transform(X_raw).astype(self.np_dtype)
        X_tensor = torch.tensor(X_scaled, dtype=self.torch_dtype, device=self.device)

        with torch.no_grad():
            y_eff_scaled = self.model_eff(X_tensor).detach().cpu().numpy()
            y_aux_scaled = self.model_aux(X_tensor).detach().cpu().numpy()

        y_eff = self.y_scaler_eff.inverse_transform(y_eff_scaled)[:, 0]
        y_aux = self.y_scaler_aux.inverse_transform(y_aux_scaled)

        Vout = y_aux[:, 0]
        Irms1 = y_aux[:, 1]
        Irms2 = y_aux[:, 2]
        Rload = X_raw[:, 1]
        Pout_calc = Vout ** 2 / np.maximum(Rload, 1e-12)
        Irms_obj = np.maximum(Irms1, Irms2)

        return {
            "efficiency": y_eff,
            "Vout": Vout,
            "Pout_calc": Pout_calc,
            "Irms1": Irms1,
            "Irms2": Irms2,
            "Irms_obj": Irms_obj,
        }

    def predict_one(self, Vin, Rload, a, b, c):
        X = np.array([[Vin, Rload, a, b, c]], dtype=self.np_dtype)
        out = self.predict_batch(X)
        return {
            "efficiency": float(out["efficiency"][0]),
            "Vout": float(out["Vout"][0]),
            "Pout_calc": float(out["Pout_calc"][0]),
            "Irms1": float(out["Irms1"][0]),
            "Irms2": float(out["Irms2"][0]),
            "Irms_obj": float(out["Irms_obj"][0]),
        }


# =========================================================
# 6. 总预测器：ReLU 稳态网络 + no_pout 代理模型
# =========================================================
class SteadyPolicyReLUNoPoutSystem:
    def __init__(self):
        self.policy = SteadyPolicyReLUPredictor(POLICY_DIR)
        use_float64 = self.policy.torch_dtype == torch.float64
        self.surrogate = NoPoutDualSurrogatePredictor(
            SURROGATE_DIR,
            use_cuda_if_available=True,
            use_float64=use_float64,
        )

    def predict_one(self, Vin, Rload, pref_eff=0.5, Vout_ref=300.0):
        angle = self.policy.predict_one(Vin, Rload, pref_eff)
        perf = self.surrogate.predict_one(
            Vin=Vin,
            Rload=Rload,
            a=angle["a"],
            b=angle["b"],
            c=angle["c"],
        )

        Pout_ref_cv = Vout_ref ** 2 / max(float(Rload), 1e-12)
        result = {}
        result.update(angle)
        result.update(perf)
        result["Vout_ref"] = float(Vout_ref)
        result["Vout_error"] = float(perf["Vout"] - Vout_ref)
        result["Pout_ref_cv"] = float(Pout_ref_cv)
        result["Pout_error_cv"] = float(perf["Pout_calc"] - Pout_ref_cv)
        return result


# =========================================================
# 7. 交互输入
# =========================================================
def interactive_predict():
    predictor = SteadyPolicyReLUNoPoutSystem()

    print("\n请输入工况参数。")
    print("输入格式: Vin Rload pref_eff")
    print("pref_eff: 0=低电流优先, 0.5=折中, 1=效率优先")
    print("输入 q 退出。\n")

    while True:
        s = input("请输入 Vin Rload pref_eff，例如 300 128.571 0.5: ").strip()
        if s.lower() in ["q", "quit", "exit"]:
            print("退出预测。")
            break

        parts = s.replace(",", " ").split()
        if len(parts) != 3:
            print("输入格式错误。正确格式示例: 300 128.571 0.5\n")
            continue

        try:
            Vin = float(parts[0])
            Rload = float(parts[1])
            pref_eff = float(parts[2])
        except ValueError:
            print("输入中存在非数字，请重新输入。\n")
            continue

        if Rload <= 0:
            print("Rload 必须大于 0。\n")
            continue

        result = predictor.predict_one(Vin=Vin, Rload=Rload, pref_eff=pref_eff, Vout_ref=300.0)

        print("\n================ 预测结果 ================")
        print("输入工况:")
        print(f"Vin             = {result['Vin']:.6f}")
        print(f"Rload           = {result['Rload']:.6f}")
        print(f"pref_eff        = {result['pref_eff']:.6f}")

        print("\nReLU 稳态网络输出最优相角:")
        print(f"a               = {result['a']:.8f}")
        print(f"b               = {result['b']:.8f}")
        print(f"c               = {result['c']:.8f}")

        print("\n代理模型预测输出:")
        print(f"efficiency      = {result['efficiency']:.8f}")
        print(f"Vout            = {result['Vout']:.8f}")
        print(f"Pout_calc       = {result['Pout_calc']:.8f}")
        print(f"Irms1           = {result['Irms1']:.8f}")
        print(f"Irms2           = {result['Irms2']:.8f}")
        print(f"Irms_obj        = {result['Irms_obj']:.8f}")

        print("\n恒压参考偏差:")
        print(f"Vout_ref        = {result['Vout_ref']:.8f}")
        print(f"Vout_error      = {result['Vout_error']:.8f}")
        print(f"Pout_ref_cv     = {result['Pout_ref_cv']:.8f}")
        print(f"Pout_error_cv   = {result['Pout_error_cv']:.8f}")
        print("========================================\n")


if __name__ == "__main__":
    interactive_predict()

    # 也可以在其他脚本中调用：
    # predictor = SteadyPolicyReLUNoPoutSystem()
    # result = predictor.predict_one(Vin=300.0, Rload=128.571, pref_eff=0.5)
    # print(result)
