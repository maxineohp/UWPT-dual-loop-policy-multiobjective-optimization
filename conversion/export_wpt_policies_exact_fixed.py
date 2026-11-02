# -*- coding: utf-8 -*-
"""Convert the two *trained WPT policy networks* to physically meaningful ONNX.

Inputs of *_full.onnx are RAW physical features, NOT standardized features.
Outputs of *_full.onnx are angles (steady) or angle increments (dynamic), in degrees.
*_core.onnx is a diagnostic raw-MLP export only: input already standardized,
output pre-Sigmoid/pre-Tanh logits. It is NOT a deployable controller.

Only load .pth/.pkl files you created and trust (PyTorch/joblib are pickle-based).
Run: python export_wpt_policies_exact_fixed.py --model-dir . --out-dir output --inspect
     python export_wpt_policies_exact_fixed.py --model-dir . --out-dir output --export
Requires: Python, torch, numpy, joblib; for --export also onnx. onnxruntime optional.
"""
from __future__ import annotations

import argparse
import inspect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import joblib

STEADY_FILES = ('steady_policy_nsga_pref.pth', 'steady_policy_input_scaler.pkl',
                'steady_policy_angle_bounds.npy')
DYNAMIC_FILES = ('dynamic_policy_best.pth', 'dynamic_policy_input_scaler.pkl')


def load_checkpoint(path: Path):
    """User-owned models ONLY: torch checkpoint deserialization can run pickle."""
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(str(path), **kwargs)


def make_core_from_saved_state(state: dict) -> nn.Sequential:
    ids = sorted(int(name.split('.')[1]) for name, weight in state.items()
                 if name.startswith('net.') and name.endswith('.weight') and weight.ndim == 2)
    expected = list(range(0, ids[-1] + 1, 2)) if ids else []
    if ids != expected:
        raise ValueError('Unsupported state layout: net.[0,2,...].weight')
    layers = []
    for i, idx in enumerate(ids):
        weight = state[f'net.{idx}.weight']
        bias = state[f'net.{idx}.bias']
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        with torch.no_grad():
            layer.weight.copy_(weight.float())
            layer.bias.copy_(bias.float())
        layers.append(layer)
        if i != len(ids) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers).eval().float()


class SteadyFull(nn.Module):
    def __init__(self, core, mean, std, lower, upper):
        super().__init__()
        self.net = core
        self.register_buffer('mean', torch.as_tensor(np.array(mean).copy(), dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(np.array(std).copy(), dtype=torch.float32))
        self.register_buffer('lower', torch.as_tensor(np.array(lower).copy(), dtype=torch.float32))
        self.register_buffer('upper', torch.as_tensor(np.array(upper).copy(), dtype=torch.float32))

    def forward(self, vin_r_pref):
        standardized = (vin_r_pref - self.mean) / self.std
        return self.lower + torch.sigmoid(self.net(standardized)) * (self.upper - self.lower)


class DynamicFull(nn.Module):
    def __init__(self, core, mean, std, du_max):
        super().__init__()
        self.net = core
        self.register_buffer('input_mean', torch.as_tensor(np.array(mean).copy(), dtype=torch.float32))
        self.register_buffer('input_std', torch.as_tensor(np.array(std).copy(), dtype=torch.float32))
        self.register_buffer('du_max', torch.as_tensor(np.array(du_max).copy(), dtype=torch.float32))

    def forward(self, raw_features_20):
        standardized = (raw_features_20 - self.input_mean) / self.input_std
        return self.du_max * torch.tanh(self.net(standardized))


def load_policies(root: Path):
    for filename in (*STEADY_FILES, *DYNAMIC_FILES):
        if not (root / filename).is_file():
            raise FileNotFoundError(f'Missing required file: {root / filename}')

    steady_ck = load_checkpoint(root / STEADY_FILES[0])
    steady_sc = joblib.load(root / STEADY_FILES[1])
    bound_file = np.load(root / STEADY_FILES[2])
    ck_bounds = np.asarray(steady_ck['angle_bounds'], dtype=np.float64)
    if bound_file.shape != (3, 2) or not np.allclose(ck_bounds, bound_file):
        raise ValueError('Steady angle bounds checkpoint vs npy mismatch!')
    if 'lower' in steady_ck['model_state_dict']:
        assert np.allclose(steady_ck['model_state_dict']['lower'].numpy(), bound_file[:, 0])
        assert np.allclose(steady_ck['model_state_dict']['upper'].numpy(), bound_file[:, 1])
    if list(steady_sc['columns']) != ['Vin', 'Rload', 'pref_eff']:
        raise ValueError('Steady input order does not match training source.')
    steady_core = make_core_from_saved_state(steady_ck['model_state_dict'])
    steady = SteadyFull(steady_core, steady_sc['mean'], steady_sc['std'],
                        bound_file[:, 0], bound_file[:, 1]).eval()

    dyn_ck = load_checkpoint(root / DYNAMIC_FILES[0])
    dyn_sc = joblib.load(root / DYNAMIC_FILES[1])
    dyn_mean = np.asarray(dyn_ck['input_mean'], dtype=np.float32)
    dyn_std = np.asarray(dyn_ck['input_std'], dtype=np.float32)
    du_max = np.asarray(dyn_ck['du_max'], dtype=np.float32)
    if not np.allclose(dyn_mean, dyn_sc['mean']) or not np.allclose(dyn_std, dyn_sc['std']):
        raise ValueError('Dynamic scaler checkpoint vs pkl mismatch!')
    if list(dyn_ck['input_names']) != list(dyn_sc['columns']):
        raise ValueError('Dynamic feature order checkpoint vs pkl mismatch!')
    for name, arr in [('input_mean', dyn_mean), ('input_std', dyn_std), ('du_max', du_max)]:
        saved = dyn_ck['model_state_dict'].get(name)
        if saved is None or not np.allclose(saved.numpy(), arr):
            raise ValueError(f'Dynamic state_dict {name} mismatch!')
    dynamic_core = make_core_from_saved_state(dyn_ck['model_state_dict'])
    dynamic = DynamicFull(dynamic_core, dyn_mean, dyn_std, du_max).eval()
    if steady_core[0].in_features != 3 or dynamic_core[0].in_features != 20:
        raise ValueError('Unexpected number of policy input features.')
    if steady_core[-1].out_features != 3 or dynamic_core[-1].out_features != 3:
        raise ValueError('Unexpected number of policy outputs.')
    return steady_ck, dyn_ck, steady_sc, dyn_sc, steady, dynamic


def steady_reference_fp64(ck, scaler, x):
    """Independent original double precision inference from SteadyAnglePolicy.forward()."""
    state = ck['model_state_dict']
    wnames = [f'net.{n}.weight' for n in [0, 2, 4, 6, 8]]
    t = (torch.tensor(x, dtype=torch.float64)
         - torch.tensor(scaler['mean'], dtype=torch.float64))
    t = t / torch.tensor(scaler['std'], dtype=torch.float64)
    for k, wname in enumerate(wnames):
        bname = wname.replace('.weight', '.bias')
        t = t @ state[wname].double().T + state[bname].double()
        if k != len(wnames) - 1:
            t = torch.relu(t)
    lower = torch.tensor(ck['angle_bounds'][:, 0], dtype=torch.float64)
    upper = torch.tensor(ck['angle_bounds'][:, 1], dtype=torch.float64)
    return (lower + torch.sigmoid(t) * (upper - lower)).numpy()


def make_dynamic_inputs(ck):
    """20-column inputs precisely following build_policy_input() of training source."""
    rows = []
    example_conditions = [
        (195., 3.1, 3.0, .04, (135., 125., 110.), (145., 131., 115.), (.1, -.2, .1), 200., 100.),
        (202., 2.5, 2.7, -.03, (170., 150., 135.), (172., 153., 140.), (0., 0., 0.), 240., 120.),
        (187., 3.5, 3.2, .10, (110., 116., 100.), (130., 130., 112.), (.7, -.4, .5), 180., 80.),
        (200., 2.9, 3.0, 0., (140., 140., 120.), (140., 140., 120.), (0., 0., 0.), 220., 110.),
    ]
    vref = float(ck['Vref'])
    for vout, ip, is_, integral, angles, targets, previous, vin, rload in example_conditions:
        diff = [tar-ang for tar, ang in zip(targets, angles)]
        features = [vout, ip, is_, vref-vout, integral,
                    *angles, *targets, *diff, *previous, vin, rload, vref]
        assert len(features) == 20
        rows.append(features)
    return np.asarray(rows, dtype=np.float32)


def export_model(module, x_example, filepath, input_name, output_name):
    kwargs = {'export_params': True, 'do_constant_folding': True,
              'opset_version': 13, 'input_names': [input_name], 'output_names': [output_name]}
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        kwargs['dynamo'] = False  # Onnx legacy graph for stable fixed-shape MCU export
    torch.onnx.export(module, x_example[:1], str(filepath), **kwargs)
    import onnx
    onnx.checker.check_model(onnx.load(str(filepath)))


def verify_onnx(path, input_data, py_output):
    try:
        import onnxruntime as ort
    except ImportError:
        print(f'[INFO] onnxruntime not installed, skipped ONNXRuntime cross-check: {path.name}')
        return None
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    # ONNX export deliberately freezes batch size to 1 (one control tick).
    # Test vectors contain multiple independent operating points, so run them
    # one at a time instead of sending an incompatible batch >1.
    outputs = []
    for k in range(len(input_data)):
        single_x = np.ascontiguousarray(input_data[k:k+1], dtype=np.float32)
        single_y = session.run(None, {session.get_inputs()[0].name: single_x})[0]
        outputs.append(single_y)
    output = np.concatenate(outputs, axis=0)
    mae = float(np.max(np.abs(py_output - output)))
    print(f'[CHECK] {path.name}: {len(input_data)} samples, ONNXRuntime vs PyTorch max_abs={mae:.8g}')
    if not np.allclose(py_output, output, atol=0.002, rtol=1e-4):
        raise AssertionError(f'ONNX output discrepancy: {path.name}')
    return mae


def json_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (tuple, list)):
        return [json_serializable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): json_serializable(v) for k, v in x.items()}
    return x


def run(args):
    root, out = Path(args.model_dir).resolve(), Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sck, dck, ssc, dsc, steady, dynamic = load_policies(root)
    sx = np.asarray([[180., 80., 0.], [200., 100., .8], [220., 110., .8],
                     [240., 130., 1.], [210., 90., .25], [175., 125., .5]], np.float32)
    dx = make_dynamic_inputs(dck)
    with torch.no_grad():
        sout = steady(torch.from_numpy(sx)).numpy()
        dout = dynamic(torch.from_numpy(dx)).numpy()
        sout_ref = steady_reference_fp64(sck, ssc, sx)
        serror = float(np.max(np.abs(sout.astype(np.float64) - sout_ref)))
        din = (torch.from_numpy(dx) - torch.tensor(dck['input_mean'])) / torch.tensor(dck['input_std'])
        draw = (dck['du_max'] * np.tanh(dynamic.net(din).numpy()))
        derror = float(np.max(np.abs(dout - draw)))
    print('STEADY: raw features Vin,Rload,pref -> angles deg')
    print('  hidden_dims:', sck['hidden_dims'], 'bounds:', sck['angle_bounds'].tolist())
    print('  input mean:', np.asarray(ssc['mean']).tolist())
    print('  input std :', np.asarray(ssc['std']).tolist())
    print('  max float64-trained vs float32-exported difference [deg]:', serror)
    print('DYNAMIC: 20 raw features -> angle increments deg')
    print('  hidden_dims:', dck['hidden_dims'], 'du_max:', np.asarray(dck['du_max']).tolist())
    print('  Ts_ctrl [s]:', dck['Ts_ctrl'], 'angle_low/high:', dck['angle_low'], dck['angle_high'])
    print('  mathematical formula check max_abs:', derror)
    print('  NOTE: dynamic increments must be added and HARD-LIMITED OUTSIDE THE NN.')
    if serror > 0.01 or derror > 1e-5:
        raise AssertionError('Source vs rebuilt forward pass differs too much.')

    np.savez(out / 'wpt_policy_test_vectors.npz',
             steady_raw_features=sx, steady_angles_float32=sout,
             steady_angles_float64_reference=sout_ref,
             dynamic_raw_features=dx, dynamic_du_float32=dout)
    metadata = {
        'precision': 'float32 full models (steady original checkpoint float64)',
        'onnx_model_type': 'floating_point_NOT_TI_NPU_QAT',
        'control_Ts_s': float(dck['Ts_ctrl']),
        'steady': {'inputs': list(ssc['columns']), 'outputs': ['a', 'b', 'c'],
                   'hidden_dims': list(sck['hidden_dims']), 'angle_bounds': sck['angle_bounds'],
                   'mean': ssc['mean'], 'std': ssc['std'],
                   'forward': 'lower+sigmoid(MLP(relu; standardized input))*(upper-lower)',
                   'fp64_to_fp32_max_abs_deg_test': serror},
        'dynamic': {'inputs': list(dck['input_names']), 'outputs': ['da', 'db', 'dc'],
                    'hidden_dims': list(dck['hidden_dims']), 'mean': dck['input_mean'],
                    'std': dck['input_std'], 'du_max_deg_each_200us': dck['du_max'],
                    'angle_low': dck['angle_low'], 'angle_high': dck['angle_high'],
                    'Vref': dck['Vref'],
                    'forward': 'du_max*tanh(MLP(relu; standardized raw_input))',
                    'integral_limits': [dck['config']['integral_min'], dck['config']['integral_max']],
                    'runtime_limits_are_model_training_not_hardware_protection': True},
        'important': [
            'Steady checkpoint angles allow 10..180 all; dynamic policy hard limits 90..180,90..180,90..150.',
            'Steady full outputs are target angles. Dynamic full outputs are increments, NOT absolute angles.',
            'For TI-F28P55 NPU, TI-NPU quantization-aware training required. Float ONNX is not NPU acceleration.',
            'MCU must recompute policy input fields exactly: voltage_error=Vref-Vout; angle_error=target-current.',
            'Controller must clip actual angles, enforce voltage/current protection, and update integral/previous_du.',
        ],
    }
    (out / 'wpt_policy_metadata.json').write_text(
        json.dumps(json_serializable(metadata), indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved test vectors and metadata in', out)
    if not args.export:
        print('Inspection completed. Add --export to write four ONNX files.')
        return

    with torch.no_grad():
        export_list = [
            ('steady_policy_full_float32.onnx', steady, sx, sout, 'steady_features_raw', 'angles_deg'),
            ('dynamic_policy_full_float32.onnx', dynamic, dx, dout, 'dynamic_features_raw', 'delta_angles_deg'),
        ]
        sc = steady.net
        dc = dynamic.net
        sc_inputs = (sx - np.asarray(ssc['mean'], np.float32)) / np.asarray(ssc['std'], np.float32)
        dc_inputs = (dx - np.asarray(dck['input_mean'], np.float32)) / np.asarray(dck['input_std'], np.float32)
        export_list.extend([
            ('steady_policy_core_float32.onnx', sc, sc_inputs, sc(torch.from_numpy(sc_inputs)).numpy(), 'steady_features_scaled', 'logits'),
            ('dynamic_policy_core_float32.onnx', dc, dc_inputs, dc(torch.from_numpy(dc_inputs)).numpy(), 'dynamic_features_scaled', 'logits'),
        ])
        for filename, model, x, y, iname, oname in export_list:
            target = out / filename
            export_model(model, torch.from_numpy(x).float(), target, iname, oname)
            verify_onnx(target, x.astype(np.float32), y.astype(np.float32))
            print('[OK] ONNX exported:', target)
    print('[DONE] All four ONNX files exported and validated.')
    print('IMPORTANT: these are FLOAT32 models; for hardware TI-NPU offload perform TI-specific QAT and retraining.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default='.', help='dir containing user-owned .pth/.pkl/.npy models')
    parser.add_argument('--out-dir', default='output', help='output directory')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--inspect', action='store_true', help='validate/produce metadata (default)')
    group.add_argument('--export', action='store_true', help='validate and export ONNX models')
    run(parser.parse_args())
