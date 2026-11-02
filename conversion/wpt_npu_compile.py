#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F28P55: check toolchain, optionally correct known ONNX rank broadcast,
compile TI QAT ONNX, and verify genuine NPU offloading.

Original QAT ONNX files and checkpoints are never modified. Only a copy of the
ONNX may have the rank of input-quantizer scalar tensors corrected.
This diagnostic change may NOT be sufficient to enable NPU acceleration.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KINDS = ('steady', 'dynamic')
DEFAULT_CGT = Path(r'C:\ti\ti_cgt_c2000_22.6.2.LTS')
ONNX_NAMES = {'steady': 'steady_policy_npu_int8.onnx',
              'dynamic': 'dynamic_policy_npu_int8.onnx'}


def pick_cgt(override: str | None) -> Path:
    choices = []
    if override:
        choices.append(Path(override))
    else:
        if os.environ.get('C2000_CGT_PATH'):
            choices.append(Path(os.environ['C2000_CGT_PATH']))
        choices += [DEFAULT_CGT]
        if os.name == 'nt' and Path(r'C:\ti').is_dir():
            choices += sorted(Path(r'C:\ti').glob('*cgt*c2000*'), reverse=True)
    details = []
    for root in choices:
        root = root.expanduser()
        exe = root / 'bin' / ('cl2000.exe' if os.name == 'nt' else 'cl2000')
        header = root / 'include' / 'stdio.h'
        details.append(f'  {root}: cl2000={exe.is_file()}, stdio.h={header.is_file()}')
        if exe.is_file() and header.is_file():
            print(f'[CGT] Selected: {root.resolve()}', flush=True)
            return root.resolve()
    raise RuntimeError('C2000 CGT not found with BOTH bin\\cl2000.exe and include\\stdio.h.\n'
                       + '\n'.join(details) + '\n'
                       'Find the compiler used by CCS and edit CGT_DIR in the BAT file. '
                       'Do NOT copy a random stdio.h from a different toolchain.')


def preflight_cgt(cgt: Path) -> None:
    """Compile a real #include <stdio.h> test using the exact TI compiler."""
    cc = cgt / 'bin' / ('cl2000.exe' if os.name == 'nt' else 'cl2000')
    inc = cgt / 'include'
    if ' ' in str(cgt):
        raise RuntimeError('CGT path contains spaces; move/select a no-space TI CGT installation. '
                           'NNC Windows documentation explicitly warns against spaces in this path.')
    # Forward slashes deliberately avoid the nested Windows quoting seen in the failing command.
    inc_flag = '-I' + inc.as_posix()
    with tempfile.TemporaryDirectory(prefix='cl2000_preflight_') as td:
        work = Path(td)
        src = work / 'probe.c'
        src.write_text('#include <stdio.h>\nint wpt_probe(void) {return 0;}\n', encoding='ascii')
        cmd = [str(cc), '-c', '-v28', '-ml', '-mt', '--abi=eabi',
               '--float_support=fpu32', '--c99', inc_flag, str(src)]
        print('[CGT TEST] ' + subprocess.list2cmdline(cmd), flush=True)
        p = subprocess.run(cmd, cwd=str(work), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, errors='replace')
        print(p.stdout[-6000:], flush=True)
        if p.returncode:
            raise RuntimeError('C2000 stdio.h PRE-FLIGHT FAILED. Wrong CGT/include root '
                               'or compiler include option. Fix before starting tvmc. '
                               f'CGT={cgt}, header={inc / "stdio.h"}')
    print('[CGT TEST] PASSED: cl2000 compiles source using <stdio.h>.', flush=True)


def find_tvmc(root: Path, supplied: str | None) -> str:
    if supplied:
        if Path(supplied).is_file():
            return str(Path(supplied).resolve())
        result = shutil.which(supplied)
        if result:
            return result
        raise FileNotFoundError(f'tvmc not found: {supplied}')
    for envdir in ('.venv', 'nnc_env', 'qat_env', 'export_env'):
        for filename in ('tvmc.exe', 'tvmc'):
            candidate = root / envdir / 'Scripts' / filename
            if candidate.is_file():
                return str(candidate.resolve())
    found = shutil.which('tvmc')
    if found:
        return found
    raise FileNotFoundError('No TI NNC tvmc found. Expected .venv\\Scripts\\tvmc.exe')


def _tensor_shape(value) -> list[int]:
    return [int(d.dim_value) for d in value.type.tensor_type.shape.dim]


def prepare_rank2_onnx(base: Path, kind: str, enable: bool = True) -> tuple[Path, dict]:
    """Normalize a *known* TI QAT input quantizer rank-4 scalar broadcast.

    The original QAT ONNX is never overwritten. Only shapes of verified scalar
    initializers may change, from [1,1,1,1] to [1,1]. Their tensor values,
    network weights and arithmetic operators are unchanged. This is an
    experiment: it does not guarantee NPU offloading.
    """
    source = base / kind / ONNX_NAMES[kind]
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f'Missing QAT ONNX: {source}. Run QAT first.')
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError('ONNX Python package is required in the NNC launcher '
                           'environment for safe rank validation. Install it in '
                           'the .venv Python used to run the compiler.') from exc

    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError(f'{kind}: expected a single ONNX input and output')
    input_dims = _tensor_shape(model.graph.input[0])
    output_dims = _tensor_shape(model.graph.output[0])
    ops = sorted({node.op_type for node in model.graph.node})
    print(f'[{kind.upper()}] input={input_dims}, output={output_dims}, ops={ops}', flush=True)
    report = {
        'source': str(source), 'source_sha256': sha256_path(source),
        'input_shape': input_dims, 'original_output_shape': output_dims,
        'modified_initializers': [], 'status': 'not_needed',
    }
    if not enable:
        report['status'] = 'disabled'
        return source, report
    if len(input_dims) != 2:
        raise RuntimeError(f'{kind}: expected rank-2 QAT input; got {input_dims}. '
                           'Refusing to guess a graph transformation.')
    if len(output_dims) == 2:
        print(f'[RANK2] {kind}: model already rank 2; no repair needed.')
        return source, report
    if (len(output_dims) != 4 or output_dims[:3] != [1, 1, 1]):
        raise RuntimeError(f'{kind}: unexpected output shape {output_dims}. '
                           'Automated repair supports only [1,1,1,C] -> [1,C].')

    # Recognize the specific quantizer chain observed in TI QAT:
    # input + oss_X.offset -> * oss_X.mult -> ... -> first MatMul.
    input_name = model.graph.input[0].name
    add_nodes = [node for node in model.graph.node
                 if node.op_type == 'Add' and input_name in node.input]
    if len(add_nodes) != 1:
        raise RuntimeError(f'{kind}: could not unambiguously locate input quantizer Add.')
    add_node = add_nodes[0]
    other_add_inputs = [name for name in add_node.input if name != input_name]
    if len(other_add_inputs) != 1:
        raise RuntimeError(f'{kind}: unusual input quantizer Add inputs.')
    offset_name = other_add_inputs[0]
    if not (offset_name.startswith('oss_') and offset_name.endswith('.offset')):
        raise RuntimeError(f'{kind}: unexpected quantizer offset {offset_name!r}.')
    multiplier_name = offset_name[:-len('.offset')] + '.mult'
    mul_nodes = [node for node in model.graph.node
                 if node.op_type == 'Mul' and add_node.output[0] in node.input]
    if len(mul_nodes) != 1 or multiplier_name not in mul_nodes[0].input:
        raise RuntimeError(f'{kind}: could not verify input quantizer multiplier {multiplier_name}.')

    initializer_map = {tensor.name: tensor for tensor in model.graph.initializer}
    # Some QAT models contain a third scalar quantizer multiplier. It is in a
    # second Mul node, after the normal ``oss_X.mult``. If we leave its rank
    # at [1,1,1,1], it broadcasts the activation back to rank 4, causing
    # /6/6.1/Clip to fail strict ONNX inference even after the first two fixes.
    shift_multiplier_name = offset_name[:-len('.offset')] + '.shift_mult'
    first_mul_output = mul_nodes[0].output[0]
    second_mul_nodes = [node for node in model.graph.node
                        if node.op_type == 'Mul' and first_mul_output in node.input]
    expected_names = [offset_name, multiplier_name]
    shift_tensor = initializer_map.get(shift_multiplier_name)
    if shift_tensor is not None:
        if len(second_mul_nodes) != 1 or shift_multiplier_name not in second_mul_nodes[0].input:
            raise RuntimeError(f'{kind}: shift multiplier {shift_multiplier_name} exists '
                               'but the expected connected Mul node was not found.')
        expected_names.append(shift_multiplier_name)
    expected = tuple(expected_names)

    # Refuse any unrecognized rank-4 quantizer constant; silently skipping it
    # would make the graph rank-4 again, as observed in the dynamic model.
    for tensor in model.graph.initializer:
        if tensor.name.startswith(offset_name[:-len('.offset')] + '.') \
                and list(tensor.dims) == [1, 1, 1, 1] and tensor.name not in expected:
            raise RuntimeError(f'{kind}: additional rank-4 input quantizer initializer '
                               f'{tensor.name}. Inspect before patching.')

    for name in expected:
        tensor = initializer_map.get(name)
        if tensor is None or list(tensor.dims) != [1, 1, 1, 1]:
            raise RuntimeError(f'{kind}: expected 4D scalar initializer {name}, '
                               f'got {None if tensor is None else list(tensor.dims)}.')
        # The initializer must represent one scalar element.
        from onnx import numpy_helper
        if numpy_helper.to_array(tensor).size != 1:
            raise RuntimeError(f'{kind}: {name} is not a scalar tensor.')
        # Rank-only correction. Tensor contents, operators and weights stay unchanged.
        tensor.ClearField('dims')
        tensor.dims.extend([1, 1])
    print(f'[RANK2] {kind}: correcting {len(expected)} scalar quantizer initializers: '
          + ', '.join(expected), flush=True)
    shape = model.graph.output[0].type.tensor_type.shape
    shape.ClearField('dim')
    for dimension in (int(output_dims[0]), int(output_dims[-1])):
        shape.dim.add().dim_value = dimension
    # TI QAT can leave stale ONNX value_info annotations on intermediate
    # activations. In the dynamic graph, /6/6.1/Clip was reported as rank 2
    # although the unmodified input quantizer broadcast produced rank 4.
    # After changing the two scalar quantizer constants to rank 2, all such
    # cached *intermediate* shapes must be recomputed from the new graph.
    # Do NOT delete graph inputs/outputs or modify weights/operation nodes.
    old_intermediates = len(model.graph.value_info)
    model.graph.ClearField('value_info')
    report['discarded_cached_intermediate_shapes'] = old_intermediates
    print(f'[RANK2] {kind}: cleared {old_intermediates} cached '
          'intermediate shape annotations prior to inference.', flush=True)

    # Keep graph.output as [1,C] so strict inference verifies that this
    # annotation really agrees with the connected operators.
    onnx.checker.check_model(model)
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        raise RuntimeError(
            f'{kind}: rank-2 shape inference still fails after invalidating '
            'cached intermediate annotations. The dynamic ONNX likely has '
            'another rank-4 broadcast or a non-scalar constant requiring '
            'inspection; do not bypass strict validation. '
            f'Underlying error: {exc}'
        ) from exc
    inferred_output = _tensor_shape(inferred.graph.output[0])
    expected_shape = [output_dims[0], output_dims[-1]]
    if inferred_output != expected_shape:
        raise RuntimeError(f'{kind}: inferred output {inferred_output} does not match '
                           f'expected rank-2 {expected_shape}')
    onnx.checker.check_model(inferred)

    destination = source.with_name(source.stem + '_rank2_fixed.onnx')
    tmp = source.with_name(source.stem + '_rank2_fixed.tmp.onnx')
    try:
        onnx.save(inferred, str(tmp))
        onnx.checker.check_model(onnx.load(str(tmp)))
        tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    report.update(status='rank2_fixed', patched=str(destination),
                  patched_sha256=sha256_path(destination),
                  modified_initializers=list(expected),
                  patched_output_shape=inferred_output)

    # Verify outputs for the original and corrected ONNX on repeatable test
    # inputs. This uses NumPy + the installed ONNX library; onnxruntime is
    # optional. If onnxruntime is installed, also run the official runtime.
    import numpy as np
    from onnx import numpy_helper

    def evaluate_small_graph(graph_model, data):
        """Reference evaluation for the small QAT graph's explicit operators.

        Unsupported operators fail closed instead of silently skipping numerical
        validation. This is NOT a hardware or NNC offloading validation.
        """
        graph = graph_model.graph
        values = {tensor.name: numpy_helper.to_array(tensor)
                  for tensor in graph.initializer}
        values[graph.input[0].name] = data
        for node in graph.node:
            ins = [values[name] for name in node.input if name]
            if node.op_type == 'Constant':
                attr = next((attr for attr in node.attribute if attr.name == 'value'), None)
                if attr is None or not attr.HasField('t'):
                    raise RuntimeError(f'{kind}: unsupported Constant at {node.name}')
                value = numpy_helper.to_array(attr.t)
            elif node.op_type == 'Identity':
                value = ins[0]
            elif node.op_type == 'Add':
                value = ins[0] + ins[1]
            elif node.op_type == 'Mul':
                value = ins[0] * ins[1]
            elif node.op_type == 'MatMul':
                value = np.matmul(ins[0], ins[1])
            elif node.op_type == 'Floor':
                value = np.floor(ins[0])
            elif node.op_type == 'Clip':
                value = np.clip(ins[0], ins[1], ins[2])
            elif node.op_type == 'Relu':
                value = np.maximum(ins[0], 0)
            else:
                raise RuntimeError(f'{kind}: reference evaluator does not support '
                                   f'{node.op_type} at {node.name}')
            if len(node.output) != 1:
                raise RuntimeError(f'{kind}: unsupported multiple outputs at {node.name}')
            values[node.output[0]] = value
        return np.asarray(values[graph.output[0].name]).reshape(-1)

    raw_model = onnx.load(str(source))
    fixed_model = onnx.load(str(destination))
    rng = np.random.default_rng(20261020)
    samples = [np.zeros(tuple(input_dims), dtype=np.float32),
               np.ones(tuple(input_dims), dtype=np.float32),
               -np.ones(tuple(input_dims), dtype=np.float32)]
    samples.extend(rng.standard_normal(tuple(input_dims)).astype(np.float32)
                   for _ in range(32))
    max_error = 0.0
    for x in samples:
        original_y = evaluate_small_graph(raw_model, x)
        corrected_y = evaluate_small_graph(fixed_model, x)
        if original_y.shape != corrected_y.shape:
            raise RuntimeError(f'{kind}: ONNX output vector shapes differ')
        max_error = max(max_error, float(np.max(np.abs(original_y - corrected_y))))
    if not np.isfinite(max_error) or max_error > 1e-5:
        raise RuntimeError(f'{kind}: rank correction changed results by {max_error}')
    report['numerical_check'] = f'passed {len(samples)} NumPy graph tests, max absolute error={max_error:.9g}'
    print(f'[RANK2] {kind}: numerical check PASS ({len(samples)} vectors); '
          f'max error {max_error:.9g}', flush=True)

    try:
        import onnxruntime as ort
    except ImportError:
        report['onnxruntime_check'] = 'unavailable (NumPy graph test passed)'
    else:
        old_session = ort.InferenceSession(str(source), providers=['CPUExecutionProvider'])
        new_session = ort.InferenceSession(str(destination), providers=['CPUExecutionProvider'])
        err = 0.0
        for x in samples:
            old = old_session.run(None, {input_name: x})[0].reshape(-1)
            new = new_session.run(None, {input_name: x})[0].reshape(-1)
            err = max(err, float(np.max(np.abs(old - new))))
        if not np.isfinite(err) or err > 1e-5:
            raise RuntimeError(f'{kind}: ONNX Runtime numerical verification failed, error={err}')
        report['onnxruntime_check'] = f'passed, max absolute error={err:.9g}'
        print(f'[RANK2] {kind}: ONNX Runtime numerical check PASS; max error {err:.9g}')

    print(f'[RANK2] {kind}: patched {expected}, '
          f'{output_dims} -> {inferred_output}; saved {destination}', flush=True)
    return destination, report


def sha256_path(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def invoke(cmd: list[str], env: dict[str, str], log: Path) -> str:
    print('\n[TVMC] ' + subprocess.list2cmdline(cmd), flush=True)
    chunks = []
    with log.open('w', encoding='utf-8') as fp:
        fp.write('[COMMAND] ' + subprocess.list2cmdline(cmd) + '\n\n')
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env=env, text=True, encoding='utf-8', errors='replace', bufsize=1) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                chunks.append(line)
                fp.write(line)
                print(line, end='', flush=True)
            rc = proc.wait()
    output = ''.join(chunks)
    if rc:
        raise RuntimeError(f'tvmc failed with code {rc}. See {log}')
    return output


def offload_count(report: str) -> int | None:
    """Parse TI's real prefixed layer summary, not just the ti-npu target setting."""
    # The console output contains "INFO:ti_mcu_nnc:" prefixes on EVERY row.
    lines = [re.sub(r'^\s*(?:INFO|DEBUG|WARNING):ti_mcu_nnc:\s*', '', row)
             for row in report.splitlines()]
    clean = '\n'.join(lines)
    match = re.search(r'Layer Patterns Offloaded(.*?)(?:Operators not Offloaded|MODEL PARTITION|$)',
                      clean, flags=re.I | re.S)
    if not match:
        return None
    valid = {'FCONV', 'GCONV', 'DWCONV', 'PWCONV', 'PWCONVRES',
             'TCONV', 'FC', 'AVGPOOL', 'MAXPOOL'}
    count = 0
    for row in match.group(1).splitlines():
        bits = row.strip().split()
        if len(bits) == 2 and bits[0].upper() in valid and bits[1].isdigit():
            count += int(bits[1])
    return count


def clear_old_outputs(out: Path, names: list[str]) -> None:
    # Don't leave stale .a/.h in CCS import if anything fails.
    ready = out / 'ccs_import'
    for name in names:
        candidate = ready / name
        if candidate.exists():
            candidate.unlink()
    for kind in KINDS:
        folder = out / kind
        for name in (f'mod_{kind}.a', f'tvmgen_{kind}.h'):
            f = folder / name
            if f.exists():
                f.unlink()


def compile_two(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parent
    base = (Path(args.model_dir).resolve() if args.model_dir else
            root / 'output' / 'lightweight_npu_pipeline_small' / 'qat_int8')
    out = (Path(args.out_dir).resolve() if args.out_dir else
           root / 'output' / 'lightweight_npu_pipeline_small' / 'npu_artifacts')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ccs_import').mkdir(parents=True, exist_ok=True)
    ready_names = [f'mod_{k}.a' for k in KINDS] + [f'tvmgen_{k}.h' for k in KINDS]
    clear_old_outputs(out, ready_names)
    sources = {}
    rank_reports = {}
    for kind in KINDS:
        try:
            sources[kind], rank_reports[kind] = prepare_rank2_onnx(
                base, kind, enable=not args.no_rank2_fix)
        except Exception as exc:
            rank_reports[kind] = {'status': 'FAILED', 'error': str(exc)}
            (out / 'rank2_repair_report.json').write_text(
                json.dumps(rank_reports, ensure_ascii=False, indent=2),
                encoding='utf-8')
            raise
    (out / 'rank2_repair_report.json').write_text(
        json.dumps(rank_reports, ensure_ascii=False, indent=2), encoding='utf-8')
    cgt = pick_cgt(args.cgt)
    preflight_cgt(cgt)
    tvmc = find_tvmc(root, args.tvmc)
    print(f'[NNC] {tvmc}\n[MODELS] {base}', flush=True)
    if args.diagnose:
        print('[DIAGNOSTIC] Toolchain preflight completed. Skipping TVMC compilation.')
        return
    env = os.environ.copy()
    env['C2000_CGT_PATH'] = str(cgt)
    env['C2000_DEVICE_FPU'] = 'fpu32'
    env['PATH'] = str(cgt / 'bin') + os.pathsep + env.get('PATH', '')
    # Avoid nested -I"..." quoting through TVMC on Windows; TI path has no spaces.
    cl_opts = ('--float_support=fpu32 --abi=eabi -O3 --opt_for_speed=5 '
               '--c99 -v28 -ml -mt --gen_func_subsections '
               f'-I{(cgt / "include").as_posix()} -I.')
    env['CL2000_OPTIONS'] = cl_opts
    cc = str((cgt / 'bin' / ('cl2000.exe' if os.name == 'nt' else 'cl2000')).resolve())
    manifest: dict[str, dict] = {}

    for kind in KINDS:
        print(f'\n========== COMPILE {kind.upper()} ==========')
        folder = out / kind
        folder.mkdir(parents=True, exist_ok=True)
        lib = folder / f'mod_{kind}.a'
        header = folder / f'tvmgen_{kind}.h'
        cmd = [tvmc, 'compile', '--target=c, ti-npu', '--target-c-mcpu=c28',
               f'--module-name={kind}', str(sources[kind]), '-o', str(lib),
               '--cross-compiler=' + cc, '--cross-compiler-options=' + cl_opts]
        output = invoke(cmd, env, out / f'{kind}_compile.log')
        offloaded = offload_count(output)
        print(f'\n[OFFLOAD] {kind} accelerated layer patterns: {offloaded}', flush=True)
        if offloaded is None:
            raise RuntimeError('Cannot verify NPU offload from the compiler report. '
                               'Treat the generated archive as NOT YET VERIFIED.')
        if offloaded == 0:
            raise RuntimeError(f'{kind}: ZERO LAYERS OFFLOADED TO NPU even after rank2 check. '
                               'The QAT ONNX compiles to CPU operations, not the desired acceleration. '
                               'See the compiler offload report and inspect FC/MatMul pattern and quantization. '
                               f'Log: {out / f"{kind}_compile.log"}')
        if not lib.is_file() or lib.stat().st_size == 0 or not header.is_file() or header.stat().st_size == 0:
            raise RuntimeError(f'{kind}: missing expected .a/.h even though tvmc returned success. '
                               f'lib={lib.exists()}, header={header.exists()}')
        manifest[kind] = {'model': str(sources[kind]), 'rank2': rank_reports[kind], 'library': str(lib),
                          'header': str(header), 'offloaded_layer_patterns': offloaded}
    for kind in KINDS:
        for ext, prefix in (('a', 'mod'), ('h', 'tvmgen')):
            filename = f'{prefix}_{kind}.{ext}'
            shutil.copy2(out / kind / filename, out / 'ccs_import' / filename)
    (out / 'compile_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('\nSUCCESS: both models compiled AND NPU offloading verified.')
    print('[CCS] ' + str(out / 'ccs_import'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cgt', help='C2000 CGT root containing bin/cl2000.exe and include/stdio.h')
    ap.add_argument('--tvmc', help='TI NNC tvmc.exe')
    ap.add_argument('--model-dir', help='directory containing steady/ and dynamic/ QAT ONNX subfolders')
    ap.add_argument('--out-dir', help='artifact output directory')
    ap.add_argument('--diagnose', action='store_true', help='preflight CGT + ONNX only, do not compile')
    ap.add_argument('--no-rank2-fix', action='store_true', help='Disable experimental QAT input-quantizer rank correction')
    args = ap.parse_args()
    try:
        compile_two(args)
        return 0
    except (OSError, RuntimeError, ValueError) as e:
        print('\nERROR: ' + str(e), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
