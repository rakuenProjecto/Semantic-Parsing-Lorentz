# MRPC Epoch 3 Task Ablation Comparison

- task: `mrpc`
- epoch: `3`
- run_tag: `glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443`
- output_root: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443`
- log_root: `logs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443`

## Summary Table

| experiment | train_acc | val_acc | val_loss | curvature_min_fraction | curvature_collapse_penalty | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 0.7017 | 0.7623 | 0.5378 | 0 | 0 | 0.981264 | -0.186066 | -0.136237 | 2.51516 |
| bert_no_jacobian_reg | 0.702 | 0.6765 | 0.6361 | 0 | 0 | 1.43096 | 0.304185 | 0.310577 | 86.7539 |
| bert_no_curvature_aux | 0.6745 | 0.6838 | 0.626 | 0 | 0.292954 | 0.14175 | -0.567593 | 0.486695 | 2.11942 |
| bert_no_aux_all | 0.6745 | 0.6838 | 0.6859 | 0 | 0 | 0.378317 | -0.311676 | -0.162193 | 1.60702 |

## Quick Interpretation

- `val_acc`는 단순 분류 정확도다.
- `curvature_min_fraction`이 1에 가까우면 곡률이 최소값으로 무너진 것이다.
- `corr_complexity_abs_c_val`이 높으면 문장 복잡도와 곡률 크기가 잘 연결된 것이다.
- `corr_complexity_jac_val`은 문장 복잡도와 Jacobian 반응의 연결 정도다.
- `jac_frob_mean`이 너무 커지면 좌표 변환 반응이 과하게 커졌을 가능성이 있다.

## Paths

### main
- output_dir: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/main`
- log_path: `logs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/main.log`
- diagnostics_path: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/main/diagnostics_epoch_3.json`
- diagnostics_exists: `True`

### bert_no_jacobian_reg
- output_dir: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_jacobian_reg`
- log_path: `logs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_jacobian_reg.log`
- diagnostics_path: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_jacobian_reg/diagnostics_epoch_3.json`
- diagnostics_exists: `True`

### bert_no_curvature_aux
- output_dir: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_curvature_aux`
- log_path: `logs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_curvature_aux.log`
- diagnostics_path: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_curvature_aux/diagnostics_epoch_3.json`
- diagnostics_exists: `True`

### bert_no_aux_all
- output_dir: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_aux_all`
- log_path: `logs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_aux_all.log`
- diagnostics_path: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_aux_all/diagnostics_epoch_3.json`
- diagnostics_exists: `True`

