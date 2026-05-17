# RTE Epoch 7 Task Ablation Comparison

- task: `rte`
- epoch: `7`
- run_tag: `glue_rte_e7_a10_isolated_n054.hpc_20260517_222501`
- output_root: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501`
- log_root: `logs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501`

## Summary Table

| experiment | train_acc | val_acc | val_loss | curvature_min_fraction | curvature_collapse_penalty | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 0.8305 | 0.5632 | 0.6947 | 0 | 0 | 1.30934 | 0.826813 | 0.610532 | 1499.56 |
| bert_no_jacobian_reg | 0.792 | 0.6065 | 0.7572 | 0 | 0 | 1.1785 | 0.724107 | 0.171118 | 7.44829 |
| bert_no_curvature_aux | 0.4928 | 0.5271 | 0.6931 | 0 | 0 | 1.93933 | 0.798007 | -0.449529 | 522061 |
| bert_no_aux_all | 0.5687 | 0.4729 | 0.6926 | 0 | 0 | 1.01973 | 0.227701 | 0.165281 | 19.1771 |

## Quick Interpretation

- `val_acc`는 단순 분류 정확도다.
- `curvature_min_fraction`이 1에 가까우면 곡률이 최소값으로 무너진 것이다.
- `corr_complexity_abs_c_val`이 높으면 문장 복잡도와 곡률 크기가 잘 연결된 것이다.
- `corr_complexity_jac_val`은 문장 복잡도와 Jacobian 반응의 연결 정도다.
- `jac_frob_mean`이 너무 커지면 좌표 변환 반응이 과하게 커졌을 가능성이 있다.

## Paths

### main
- output_dir: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/main`
- log_path: `logs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/main.log`
- diagnostics_path: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/main/diagnostics_epoch_7.json`
- diagnostics_exists: `True`

### bert_no_jacobian_reg
- output_dir: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_jacobian_reg`
- log_path: `logs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_jacobian_reg.log`
- diagnostics_path: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_jacobian_reg/diagnostics_epoch_7.json`
- diagnostics_exists: `True`

### bert_no_curvature_aux
- output_dir: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_curvature_aux`
- log_path: `logs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_curvature_aux.log`
- diagnostics_path: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_curvature_aux/diagnostics_epoch_7.json`
- diagnostics_exists: `True`

### bert_no_aux_all
- output_dir: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_aux_all`
- log_path: `logs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_aux_all.log`
- diagnostics_path: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_aux_all/diagnostics_epoch_7.json`
- diagnostics_exists: `True`

