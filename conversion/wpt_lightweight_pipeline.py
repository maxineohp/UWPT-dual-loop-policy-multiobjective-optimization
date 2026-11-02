#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a small steady WPT model and a newly trained small DPC dynamic model.

IMPORTANT:
  1. Selects both checkpoints ONLY from lightweight_small subdirectories.
  2. Reuses the local trusted export/QAT scripts already present in the project.
  3. Performs TI NPU QAT, checks/optionally repairs the known ONNX scalar broadcast
     rank issue, then invokes NNC.
  4. Only successful, nonzero NPU offload produces CCS-ready .a/.h files.

Uses standard Python libraries only in the orchestration process.
Only open/load checkpoints (.pth/.pkl) that YOU trained and trust.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
LIGHT = ROOT / 'lightweight_small'
PIPELINE_OUT = ROOT / 'output' / 'lightweight_npu_pipeline_small'
STAGE = PIPELINE_OUT / 'source_flat'
QAT_OUT = PIPELINE_OUT / 'qat_int8'
COMPILED_OUT = PIPELINE_OUT / 'npu_artifacts'

# STRICT source directories: no fallback to legacy student or full-size models.
SOURCE_CANDIDATES = {
    'steady': [LIGHT / 'steady_policy_nsga_pref_no_pout_output'],
    'dynamic': [LIGHT / 'dynamic_policy_small_dpc_output'],
}

REQUIRED = {
    'steady': ['steady_policy_nsga_pref.pth', 'steady_policy_input_scaler.pkl',
               'steady_policy_angle_bounds.npy'],
    'dynamic': ['dynamic_policy_best.pth', 'dynamic_policy_input_scaler.pkl'],
}
SCRIPTS = ('export_wpt_policies_exact_fixed.py', 'qat_wpt_dual_policy_ti150_fixed.py',
           'wpt_npu_compile.py')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def locate_sources() -> dict[str, Path]:
    selected = {}
    for kind, candidates in SOURCE_CANDIDATES.items():
        files = REQUIRED[kind]
        found = next((directory for directory in candidates
                      if all((directory / name).is_file() for name in files)), None)
        if found is None:
            suggestions = '\n'.join('  ' + str(directory) for directory in candidates)
            raise FileNotFoundError(
                f'Cannot locate {kind} student model. Expected ALL {files} in one of:\n'
                + suggestions + '\nDo not point to the full-size original by accident.')
        selected[kind] = found
    return selected


def print_sources() -> dict[str, dict]:
    print(f'Project: {ROOT}')
    print(f'Lightweight models: {LIGHT}')
    missing = []
    manifest = {}
    sources = locate_sources()
    for kind, source_dir in sources.items():
        print(f'\n[{kind.upper()}] {source_dir}')
        for filename in REQUIRED[kind]:
            file = source_dir / filename
            if not file.is_file() or file.stat().st_size == 0:
                missing.append(str(file))
                print('  MISSING:', filename)
            else:
                info = {'source': str(file), 'bytes': file.stat().st_size,
                        'sha256': sha256_file(file)}
                manifest[filename] = info
                print(f'  OK {filename}: {info["bytes"]:,} bytes, sha256 {info["sha256"][:16]}...')
    if missing:
        raise RuntimeError('Missing lightweight source file(s):\n  ' + '\n  '.join(missing))
    print('\n[CHECK] All 5 lightweight input files are present.')
    print('[CHECK] Both sources are taken only from lightweight_small.')
    return manifest


def check_scripts() -> None:
    missing = [str(ROOT / name) for name in SCRIPTS if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError('Required local script(s) not found:\n  ' + '\n  '.join(missing))


def prepare_stage(manifest: dict[str, dict]) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    for name, info in manifest.items():
        src = Path(info['source'])
        dest = STAGE / name
        if not dest.exists() or sha256_file(dest) != info['sha256']:
            shutil.copy2(src, dest)
        if sha256_file(dest) != info['sha256']:
            raise RuntimeError(f'Staged file checksum mismatch: {dest}')
    report = {
        'description': 'Newly trained small DPC dynamic policy and small steady policy for TI NPU QAT',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'files': manifest,
        'steady_network_input': '3 normalized features; QAT expands first dense layer to 16',
        'dynamic_network_input': '20 normalized features',
        'network_outputs': '3 steady logits and 3 dynamic logits; MCU does physical conversion',
    }
    (STAGE / 'source_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\n[STAGING] Copied/checksummed 5 files into:')
    print(' ', STAGE)
    print('[STAGING] Source provenance:', STAGE / 'source_manifest.json')


def run_checked(cmd: list[str], label: str, log: Path) -> None:
    print(f'\n========== {label} ==========')
    print('Command:', subprocess.list2cmdline(cmd), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as writer:
        writer.write('Command: ' + subprocess.list2cmdline(cmd) + '\n\n')
        with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True,
                              encoding='utf-8', errors='replace', bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='', flush=True)
                writer.write(line)
            code = process.wait()
    if code:
        raise RuntimeError(f'{label} failed (exit code={code}). See log: {log}')


def qat_python_from_args(arg: str | None) -> Path:
    if arg:
        path = Path(arg)
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f'QAT Python executable was not found: {path}')
    candidates = [ROOT / 'qat_env' / 'Scripts' / 'python.exe',
                  ROOT / '.venv' / 'Scripts' / 'python.exe',
                  ROOT / 'export_env' / 'Scripts' / 'python.exe']
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        'No Python executable found under qat_env/.venv/export_env. '
        'Use --qat-python "D:\\...\\qat_env\\Scripts\\python.exe".'
    )


def check_qat_dependency(py: Path) -> None:
    # Check imports in the environment that will actually be used for QAT.
    code = 'import torch, numpy, joblib, onnx; import tinyml_torchmodelopt.quantization; print("TI QAT import OK; torch", torch.__version__)'
    run_checked([str(py), '-c', code], 'Verify QAT dependencies',
                STAGE.parent / 'qat_dependencies.log')


def verify_dynamic_three_degrees(py: Path) -> None:
    # The checkpoint must describe an actual 3-degree student; changing a scalar
    # at deployment alone does NOT make a 20-degree policy a 3-degree policy.
    code = (
        'import sys, torch, numpy as np; '
        'c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); '
        'v=c.get("du_max",c.get("config",{}).get("du_max")); '
        'assert v is not None, "dynamic checkpoint has no du_max metadata"; '
        'a=np.asarray(v,dtype=float).reshape(-1); '
        'assert a.size==3 and np.allclose(a,[3,3,3],atol=1e-5), '
        'f"checkpoint du_max must equal [3,3,3], got {a}"; '
        'print("[DPC] Verified dynamic du_max:",a.tolist())'
    )
    run_checked([str(py), '-c', code, str(STAGE / 'dynamic_policy_best.pth')],
                'Verify dynamic student 3-degree scale',
                PIPELINE_OUT / 'verify_dpc_limit.log')


def current_source_hashes() -> dict[str, str]:
    return {name: sha256_file(STAGE / name)
            for filenames in REQUIRED.values() for name in filenames}


def write_qat_provenance() -> None:
    data = {'source_hashes': current_source_hashes(),
            'onnx_hashes': {
                kind: sha256_file(QAT_OUT / kind / f'{kind}_policy_npu_int8.onnx')
                for kind in ('steady', 'dynamic')},
            'generated_utc': datetime.now(timezone.utc).isoformat()}
    (QAT_OUT / 'dpc_qat_provenance.json').write_text(
        json.dumps(data, indent=2), encoding='utf-8')


def verify_qat_provenance() -> None:
    marker = QAT_OUT / 'dpc_qat_provenance.json'
    if not marker.is_file():
        raise RuntimeError('Missing DPC QAT provenance. Run --mode all or --mode qat first. '
                           'Refusing to compile possibly stale ONNX files.')
    info = json.loads(marker.read_text(encoding='utf-8'))
    if info.get('source_hashes') != current_source_hashes():
        raise RuntimeError('DPC source model/scaler changed since QAT. '
                           'Run --mode all to regenerate from the current weights.')
    for kind in ('steady', 'dynamic'):
        file = QAT_OUT / kind / f'{kind}_policy_npu_int8.onnx'
        if not file.is_file() or sha256_file(file) != info.get('onnx_hashes',{}).get(kind):
            raise RuntimeError(f'{kind} QAT ONNX missing or changed. Run --mode all.')


def do_qat(py: Path, mode: str, epochs: int, samples: int, batch_size: int,
           lr: float, inspect_only: bool = False) -> None:
    check_qat_dependency(py)
    verify_dynamic_three_degrees(py)
    inspect_cmd = [str(py), str(ROOT / 'qat_wpt_dual_policy_ti150_fixed.py'),
                   '--mode', 'inspect', '--model-dir', str(STAGE)]
    run_checked(inspect_cmd, 'Verify rebuilt lightweight networks',
                STAGE.parent / 'inspect_networks.log')
    if inspect_only:
        return
    # Remove old manifest/ONNX so a failed QAT cannot accidentally be compiled.
    old_marker = QAT_OUT / 'dpc_qat_provenance.json'
    old_marker.unlink(missing_ok=True)
    for kind in ('steady', 'dynamic'):
        (QAT_OUT / kind / f'{kind}_policy_npu_int8.onnx').unlink(missing_ok=True)
        (QAT_OUT / kind / f'{kind}_policy_npu_int8_rank2_fixed.onnx').unlink(missing_ok=True)
    # A failed QAT attempt must not leave a stale result in the CCS-ready folder.
    ready = COMPILED_OUT / 'ccs_import'
    for name in ('mod_steady.a', 'mod_dynamic.a',
                 'tvmgen_steady.h', 'tvmgen_dynamic.h'):
        (ready / name).unlink(missing_ok=True)
    (COMPILED_OUT / 'compile_manifest.json').unlink(missing_ok=True)
    cmd = [str(py), str(ROOT / 'qat_wpt_dual_policy_ti150_fixed.py'),
           '--mode', mode, '--model-dir', str(STAGE), '--out-dir', str(QAT_OUT),
           '--epochs', str(epochs), '--samples', str(samples),
           '--batch-size', str(batch_size), '--lr', str(lr)]
    run_checked(cmd, f'Run TI QAT on LIGHTWEIGHT models ({mode})',
                STAGE.parent / 'qat_train.log')
    kinds = ('steady', 'dynamic') if mode == 'both' else (mode,)
    for kind in kinds:
        onnx_path = QAT_OUT / kind / f'{kind}_policy_npu_int8.onnx'
        if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
            raise RuntimeError(f'QAT returned success, but ONNX missing or empty: {onnx_path}')
        print(f'[OK] {kind.upper()} INT8 ONNX:', onnx_path)
    # Only a complete BOTH-model QAT gets a provenance manifest.
    if mode == 'both':
        write_qat_provenance()


def do_compile(cgt: str | None, tvmc: str | None) -> None:
    verify_qat_provenance()
    # Require BOTH real QAT files, not legacy FLOAT32 models from root/output.
    for kind in ('steady', 'dynamic'):
        p = QAT_OUT / kind / f'{kind}_policy_npu_int8.onnx'
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(f'Cannot compile: required QAT ONNX missing: {p}\n'
                                    'First run --mode qat or --mode all.')
    driver = ROOT / 'wpt_npu_compile.py'
    # Run using an existing, available Python only to drive subprocesses.
    cmd = [sys.executable, str(driver),
           '--model-dir', str(QAT_OUT), '--out-dir', str(COMPILED_OUT)]
    if cgt:
        cmd += ['--cgt', cgt]
    if tvmc:
        cmd += ['--tvmc', tvmc]
    run_checked(cmd, 'Verify ONNX shapes, optionally repair rank and compile F28P55 NPU libraries',
                STAGE.parent / 'nnc_compile.log')
    ready = COMPILED_OUT / 'ccs_import'
    expected = ('mod_steady.a', 'tvmgen_steady.h',
                'mod_dynamic.a', 'tvmgen_dynamic.h')
    missing = [file for file in expected if not (ready / file).is_file()
               or (ready / file).stat().st_size == 0]
    if missing:
        raise RuntimeError('NNC returned success, but required compiled files missing: ' + str(missing))
    print('\n[REPORT]', COMPILED_OUT / 'rank2_repair_report.json')
    print('SUCCESS. NNC artifacts have verified nonzero NPU offload:')
    for file in expected:
        print(' ', ready / file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'inspect', 'qat', 'compile', 'all'), default='all')
    parser.add_argument('--qat-python', help='Explicit Python executable for QAT, e.g. qat_env\\Scripts\\python.exe')
    parser.add_argument('--cgt', help='C2000 CGT installation directory')
    parser.add_argument('--tvmc', help='TI NNC tvmc.exe path')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--samples', type=int, default=16000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 100 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs >=1, samples >=100, batch-size >=1, and lr >0 are required')
    print('F28P55 NEW SMALL WPT policies -> TI QAT -> rank check -> NNC -> .a / .h')
    manifest = print_sources()
    check_scripts()
    prepare_stage(manifest)
    if args.mode == 'check':
        print('\nCHECK COMPLETE (no network was trained or compiled).')
        return 0
    if args.mode in ('inspect', 'qat', 'all'):
        py = qat_python_from_args(args.qat_python)
        print('\n[QAT ENV]', py)
        do_qat(py, 'both', args.epochs, args.samples, args.batch_size,
               args.lr, inspect_only=args.mode == 'inspect')
    if args.mode in ('compile', 'all'):
        do_compile(args.cgt, args.tvmc)
    if args.mode in ('qat', 'inspect'):
        print('\nNext step: use --mode compile to produce .a and .h.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print('\nERROR:', exc, file=sys.stderr)
        sys.exit(1)
