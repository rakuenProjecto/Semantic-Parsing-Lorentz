# MRPC Epoch 7 Task Ablation Comparison

- task: `mrpc`
- epoch: `7`
- run_tag: `glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443`
- output_root: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443`
- log_root: `logs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443`

## Summary Table

| experiment | train_acc | val_acc | val_loss | curvature_min_fraction | curvature_collapse_penalty | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 0.8539 | 0.8137 | 0.4822 | 0 | 0 | 1.22229 | 0.196877 | -0.111866 | 1.62048 |
| bert_no_jacobian_reg | 0.907 | 0.7966 | 0.5041 | 0 | 0 | 0.99597 | 0.164673 | 0.220742 | 3620.4 |
| bert_no_curvature_aux | 0.6745 | 0.6838 | 0.625 | 0 | 0.628522 | 0.0914412 | -0.290161 | -0.293492 | 6.16236 |
| bert_no_aux_all | 0.6745 | 0.6838 | 0.6773 | 0 | 0 | 0.37865 | -0.311822 | -0.161933 | 1.60517 |

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
- diagnostics_path: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/main/diagnostics_epoch_7.json`
- diagnostics_exists: `True`

### bert_no_jacobian_reg
- output_dir: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_jacobian_reg`
- log_path: `logs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_jacobian_reg.log`
- diagnostics_path: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_jacobian_reg/diagnostics_epoch_7.json`
- diagnostics_exists: `True`

### bert_no_curvature_aux
- output_dir: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_curvature_aux`
- log_path: `logs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_curvature_aux.log`
- diagnostics_path: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_curvature_aux/diagnostics_epoch_7.json`
- diagnostics_exists: `True`

### bert_no_aux_all
- output_dir: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_aux_all`
- log_path: `logs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_aux_all.log`
- diagnostics_path: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443/bert_no_aux_all/diagnostics_epoch_7.json`
- diagnostics_exists: `True`

