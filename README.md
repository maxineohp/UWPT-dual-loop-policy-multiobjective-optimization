# UWPT neural network and multi-objective optimization code

This repository contains the **source scripts** for an underwater wireless power transfer (UWPT) workflow: surrogate modeling, NSGA-II preference-label generation, steady and dynamic policy training, optional model distillation, and conversion toward the TI F28P55 NPU toolchain.

**Code only:** no training data, PLECS circuits, trained models, weights, scalers, checkpoints, generated headers/libraries, conversion outputs, or experiment results are included. The scripts therefore cannot reproduce the original experiments end to end from this repository alone.

## Contents

| Directory | Purpose |
| --- | --- |
| `training/` | Train steady and dynamic surrogates and policies; generate labels and trajectories; distill and fine-tune small policies; export PLECS C-Script headers from user-supplied checkpoints. |
| `conversion/` | Inspect/export trained policies to ONNX, run TI-oriented quantization-aware training, and compile compatible models for F28P55 NPU integration. |

## Suggested workflow

1. Prepare a compatible steady-state CSV for `training/train_surrogate_no_pout_new_columns.py`. This script trains efficiency and auxiliary surrogate networks.
2. With the resulting **user-supplied** surrogate checkpoints and scalers, run `training/generate_nsga_pref_labels_200V_R100pm30_parallel.py`, then `training/train_steady_policy_from_nsga_labels_200V_Vin180_240_R80_130.py` to train a steady policy. Check the configuration dictionaries in both scripts before running them; their filenames do not necessarily reflect every current parameter value.
3. Prepare compatible trajectory CSVs. `training/train_dynamic_surrogate_one_step.py` and `training/train_dynamic_surrogate_pretrain_then_multistep.py` provide alternative dynamic surrogate training paths. `training/train_dynamic_policy_small_dpc.py` trains a dynamic policy using compatible surrogate and steady-policy artifacts.
4. Optionally use `training/distill_wpt_small_policies_excel.py` and `training/finetune_small_dynamic_dpc.py` to produce smaller policies. `training/export_plecs_headers.py` generates C-Script headers containing model parameters; **do not commit those generated headers** if weights must remain private.
5. Place your own compatible steady and dynamic policy checkpoints, scalers, and bounds where the conversion scripts expect them. `conversion/export_wpt_policies_exact_fixed.py` inspects/exports policies; `conversion/qat_wpt_dual_policy_ti150_fixed.py` performs TI-oriented QAT; `conversion/wpt_npu_compile.py` checks the toolchain and compiles QAT ONNX models. `conversion/wpt_lightweight_pipeline.py` orchestrates the small-policy path and expects its private inputs under `conversion/lightweight_small/`.

Each script contains its own path assumptions, required CSV columns, model format, and options. Review those settings and run scripts from the appropriate working directory. The `conversion/` scripts also require a compatible TI NNC/QAT installation and C2000 compiler, which are not redistributed here. Generated `.a` and `.h` files should be integrated into CCS only after numerical, timing, and hardware-safety validation.

## Python dependencies

The training scripts were written for a Python 3.11 environment. Install the general packages listed in `requirements.txt` in a new virtual environment. The TI QAT package (`tinyml_torchmodelopt`) and NNC compiler must be obtained and installed separately from TI under their own terms. Package versions and the complete workflow have **not** been validated in a clean environment with the private assets omitted.

## Privacy and reproducibility

The `.gitignore` blocks common training data, checkpoints, scalers, PLECS models, generated model headers, NPU binaries, and result folders. It does not replace review of each commit: check the file list before adding new files. PyTorch checkpoints and joblib/pickle files should be loaded only from trusted sources.

## License

The source code and documentation in this repository are provided under the [MIT License](LICENSE). No license is granted here for omitted models, weights, data, PLECS circuits, TI tools, or other third-party components.
