# RTE Epoch 3 Task Ablation Comparison

- task: `rte`
- epoch: `3`
- run_tag: `glue_rte_e7_a10_isolated_n054.hpc_20260517_222501`
- output_root: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501`
- log_root: `logs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501`

## Summary Table

| experiment | train_acc | val_acc | val_loss | curvature_min_fraction | curvature_collapse_penalty | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 0.5052 | 0.4874 | 0.6923 | 0 | 0 | 1.70174 | 0.827521 | 0.795075 | 104714 |
| bert_no_jacobian_reg | 0.5631 | 0.5126 | 0.6925 | 0 | 0 | 1.28381 | 0.78518 | -0.342064 | 12.9128 |
| bert_no_curvature_aux | 0.4984 | 0.4729 | 0.6932 | 0 | 0 | 1.70865 | 0.763911 | -0.27248 | 790299 |
| bert_no_aux_all | 0.4892 | 0.5271 | 0.693 | 0 | 0 | 0.919915 | -0.177037 | -0.249882 | 9.48645 |

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
- diagnostics_path: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/main/diagnostics_epoch_3.json`
- diagnostics_exists: `True`

### bert_no_jacobian_reg
- output_dir: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_jacobian_reg`
- log_path: `logs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_jacobian_reg.log`
- diagnostics_path: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_jacobian_reg/diagnostics_epoch_3.json`
- diagnostics_exists: `True`

### bert_no_curvature_aux
- output_dir: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_curvature_aux`
- log_path: `logs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_curvature_aux.log`
- diagnostics_path: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_curvature_aux/diagnostics_epoch_3.json`
- diagnostics_exists: `True`

### bert_no_aux_all
- output_dir: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_aux_all`
- log_path: `logs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_aux_all.log`
- diagnostics_path: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501/bert_no_aux_all/diagnostics_epoch_3.json`
- diagnostics_exists: `True`

