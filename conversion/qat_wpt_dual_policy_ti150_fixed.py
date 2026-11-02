# -*- coding: utf-8 -*-
"""TI F28P55 dual WPT policy QAT: use only trusted, user-trained checkpoint files.

Training is teacher-student distillation of the *MLP logits* with TI-NPU QAT.
At deployment, CPU performs physical-feature preprocessing and final
sigmoid/tanh scaling + angle/slew/current protection; the full MLP is targeted
at NPU. The steady model is zero-padded from 3 to 16 input features.

This script uses the pre-existing `export_wpt_policies_exact_fixed.py` loader.
Requires: torch, numpy, joblib, onnx, tinyml_torchmodelopt.
The TI QAT path cannot be tested here without TI's QAT package; --inspect
is independently tested with the user's checkpoints. Do not use unvalidated
models in live power electronics control.

Usage:
  python qat_wpt_dual_policy_ti150_fixed.py --mode inspect --model-dir .
  python qat_wpt_dual_policy_ti150_fixed.py --mode both --model-dir . --out-dir output/qat --epochs 12
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from export_wpt_policies_exact_fixed import load_policies


def expand_steady_for_npu(steady_core):
    """Mathematically exact FP32 expansion: f([x0,x1,x2,0,..,0]) == f([x0,x1,x2])."""
    model = copy.deepcopy(steady_core).eval()
    first = model[0]
    assert isinstance(first, nn.Linear) and first.in_features == 3
    wider = nn.Linear(16, first.out_features)
    with torch.no_grad():
        wider.weight.zero_()
        wider.weight[:, :3].copy_(first.weight)
        wider.bias.copy_(first.bias)
    model[0] = wider
    return model


def make_steady_raw(rng, count):
    x = np.column_stack([
        rng.uniform(170., 250., count),
        rng.uniform(70., 130., count),
        rng.uniform(0., 1., count),
    ]).astype(np.float32)
    # Emphasize the expected real operating window; never exceed model's training range.
    core = int(count * .65)
    x[:core, 0] = rng.uniform(180., 240., core)
    x[:core, 1] = rng.uniform(80., 130., core)
    return x


def make_dynamic_raw(rng, count, ck):
    """Construct physically consistent 20-feature rows, NOT 20 independent draws.

    This is an initial teacher distillation dataset only. Replace/augment with
    measured or simulated closed-loop samples before validating a control deployment.
    """
    vref = float(ck['Vref'])
    Vout = rng.uniform(145., 260., count).astype(np.float32)
    Ip = rng.uniform(.1, 5.6, count).astype(np.float32)
    Is = rng.uniform(.1, 5.1, count).astype(np.float32)
    integral = rng.uniform(-.5, .5, count).astype(np.float32)
    low = np.asarray(ck['angle_low'], dtype=np.float32)
    high = np.asarray(ck['angle_high'], dtype=np.float32)
    current = rng.uniform(low, high, (count, 3)).astype(np.float32)
    target = np.clip(current + rng.normal(0., [12., 12., 9.], (count, 3)), low, high).astype(np.float32)
    prev = rng.uniform(-np.asarray(ck['du_max']), np.asarray(ck['du_max']), (count, 3)).astype(np.float32)
    vin = rng.uniform(180., 250., count).astype(np.float32)
    rload = rng.uniform(80., 130., count).astype(np.float32)
    vref_col = np.full(count, vref, dtype=np.float32)
    return np.column_stack([
        Vout, Ip, Is, vref_col - Vout, integral,
        current, target, target - current, prev,
        vin, rload, vref_col
    ]).astype(np.float32)


def prep_features(name, count, seed, ck, scaler):
    rng = np.random.default_rng(seed)
    if name == 'steady':
        raw = make_steady_raw(rng, count)
        mean = np.asarray(scaler['mean'], dtype=np.float32)
        std = np.asarray(scaler['std'], dtype=np.float32)
        norm = (raw - mean) / std
        norm = np.pad(norm, ((0, 0), (0, 13)), constant_values=0.)
    else:
        raw = make_dynamic_raw(rng, count, ck)
        mean = np.asarray(ck['input_mean'], dtype=np.float32)
        std = np.asarray(ck['input_std'], dtype=np.float32)
        norm = (raw - mean) / std
    assert np.all(np.isfinite(norm))
    return torch.from_numpy(np.ascontiguousarray(norm, dtype=np.float32))


def check_initial_equivalence(name, teacher, student, ck, scaler):
    xx = prep_features(name, 512, 1234, ck, scaler)
    original_input = xx[:, :3] if name == 'steady' else xx
    with torch.no_grad():
        orig = teacher(original_input)
        expanded = student(xx)
    max_err = float((orig - expanded).abs().max())
    print(f'[{name}] FP32 network restoration/zero-padding max logit error: {max_err:.9g}')
    if max_err > 1e-5:
        raise RuntimeError(f'{name} network restoration or padding error')


def physical_error(name, out1, out2, ck):
    if name == 'steady':
        lo = torch.as_tensor(np.asarray(ck['angle_bounds'])[:, 0], dtype=torch.float32)
        hi = torch.as_tensor(np.asarray(ck['angle_bounds'])[:, 1], dtype=torch.float32)
        to_phys = lambda z: lo + torch.sigmoid(z) * (hi - lo)
    else:
        du = torch.as_tensor(ck['du_max'], dtype=torch.float32)
        to_phys = lambda z: du * torch.tanh(z)
    return float((to_phys(out1) - to_phys(out2)).abs().max())


def train_one(name, teacher, student, ck, scaler, args):
    try:
        import tinyml_torchmodelopt.quantization as tinpu_quantization
        from tinyml_torchmodelopt.quantization.common import TinyMLQConfigType
    except Exception as ex:
        raise RuntimeError('Cannot import TI-NPU QAT. Install tinyml_torchmodelopt in this Python environment.') from ex

    n = args.samples
    xtrain = prep_features(name, n, 20261018, ck, scaler)
    xval = prep_features(name, max(1024, n // 10), 20261019, ck, scaler)
    with torch.no_grad():
        # Target is float FP32 teacher logits, not random labels.
        target_train = teacher(xtrain[:, :3] if name == 'steady' else xtrain).detach().clone()
        target_val = teacher(xval[:, :3] if name == 'steady' else xval).detach().clone()

    train_data = DataLoader(TensorDataset(xtrain, target_train), batch_size=args.batch_size, shuffle=True)
    # TI torchmodelopt 1.5.0 requires a TinyMLQConfigType *instance*.
    # Disable automatic mixed-precision search for deterministic W8A8 QAT.
    qconfig = TinyMLQConfigType(
        weight_bitwidth=8,
        activation_bitwidth=8,
        auto_quantization=False,
    )
    qat = tinpu_quantization.TINPUTinyMLQATFxModule(
        student.train(),
        qconfig_type=qconfig,
        example_inputs=xtrain[:1].contiguous(),
        total_epochs=args.epochs,
        output_int=True,
    )
    optimizer = torch.optim.Adam(qat.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    for epoch in range(args.epochs):
        qat.train()
        running, num = 0., 0
        for x, y in train_data:
            pred = qat(x)
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{name} QAT non-finite loss at epoch {epoch+1}')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat.parameters(), 2.)
            optimizer.step()
            running += loss.item() * x.shape[0]
            num += x.shape[0]
        qat.eval()
        with torch.no_grad():
            pred_val = torch.cat([qat(x) for x in xval.split(256)], dim=0)
            deg_err = physical_error(name, pred_val, target_val, ck)
            mean_err = float((pred_val - target_val).abs().mean())
        print(f'[{name}] epoch {epoch+1:02d}/{args.epochs}, loss={running/num:.6g}, '
              f'val_mean_logit_abs={mean_err:.6g}, val_max_physical_deg={deg_err:.6g}')

    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    # Save a comparison report before integer conversion, when dequantization
    # semantics and output units are still directly comparable.
    report = dict(name=name, qat_epochs=args.epochs, source='pretrained .pth',
                  max_physical_error_degree_fakequant=deg_err,
                  mean_logit_error_fakequant=mean_err,
                  input_dimensions=16 if name == 'steady' else 20,
                  final_output_transforms_on_cpu='sigmoid angle map' if name == 'steady' else 'tanh du_max',
                  note='synthetic teacher-student QAT; verify on real closed-loop trajectories before power test')
    (out / 'qat_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    qat.eval()
    # TI official flow: qat.convert(), then export qat.module to ONNX.
    qat = qat.convert()
    dummy = xval[:1].contiguous()
    output_onnx = out / f'{name}_policy_npu_int8.onnx'
    # TI's wrapper export preserves its NPU-specific integer graph conversion.
    # Use simplify=False for initial testing to isolate quantizer/compiler errors.
    qat.export(
        dummy,
        str(output_onnx),
        opset_version=17,
        simplify=False,
        input_names=['npu_input'],
        output_names=['npu_output'],
    )
    import onnx
    onnx.checker.check_model(str(output_onnx))
    print(f'[{name}] TI-NPU quantized ONNX generated: {output_onnx}')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['inspect', 'steady', 'dynamic', 'both'], default='inspect')
    ap.add_argument('--model-dir', default='.')
    ap.add_argument('--out-dir', default='output/qat')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--samples', type=int, default=16000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-5)
    args = ap.parse_args()
    torch.manual_seed(42)
    sc, dc, ss, ds, steady_full, dynamic_full = load_policies(Path(args.model_dir))
    steady_teacher = steady_full.net.eval()
    dynamic_teacher = dynamic_full.net.eval()
    steady_npu = expand_steady_for_npu(steady_teacher)
    dynamic_npu = copy.deepcopy(dynamic_teacher).eval()
    check_initial_equivalence('steady', steady_teacher, steady_npu, sc, ss)
    check_initial_equivalence('dynamic', dynamic_teacher, dynamic_npu, dc, ds)
    if args.mode == 'inspect':
        print('Model reconstruction checked. Next: install TI QAT and run --mode both.')
        return
    models = {'steady': (steady_teacher, steady_npu, sc, ss),
              'dynamic': (dynamic_teacher, dynamic_npu, dc, ds)}
    for name in ('steady', 'dynamic'):
        if args.mode in (name, 'both'):
            train_one(name, *models[name], args)


if __name__ == '__main__':
    main()
