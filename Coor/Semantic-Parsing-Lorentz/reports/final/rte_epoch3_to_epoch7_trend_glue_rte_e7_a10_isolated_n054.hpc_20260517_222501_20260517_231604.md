# RTE Epoch 3 → Epoch 7 Trend Comparison

- task: `rte`
- run_tag: `glue_rte_e7_a10_isolated_n054.hpc_20260517_222501`
- output_root: `outputs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501`
- log_root: `logs_task_isolated/glue_rte_e7_a10_isolated_n054.hpc_20260517_222501`

## Epoch Final Values

| experiment | epoch | train_acc | val_acc | val_loss | curvature_min_fraction | curvature_collapse_penalty | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 3 | 0.5052 | 0.4874 | 0.6923 | 0 | 0 | 1.70174 | 0.827521 | 0.795075 | 104714 |
| main | 7 | 0.8305 | 0.5632 | 0.6947 | 0 | 0 | 1.30934 | 0.826813 | 0.610532 | 1499.56 |
| bert_no_jacobian_reg | 3 | 0.5631 | 0.5126 | 0.6925 | 0 | 0 | 1.28381 | 0.78518 | -0.342064 | 12.9128 |
| bert_no_jacobian_reg | 7 | 0.792 | 0.6065 | 0.7572 | 0 | 0 | 1.1785 | 0.724107 | 0.171118 | 7.44829 |
| bert_no_curvature_aux | 3 | 0.4984 | 0.4729 | 0.6932 | 0 | 0 | 1.70865 | 0.763911 | -0.27248 | 790299 |
| bert_no_curvature_aux | 7 | 0.4928 | 0.5271 | 0.6931 | 0 | 0 | 1.93933 | 0.798007 | -0.449529 | 522061 |
| bert_no_aux_all | 3 | 0.4892 | 0.5271 | 0.693 | 0 | 0 | 0.919915 | -0.177037 | -0.249882 | 9.48645 |
| bert_no_aux_all | 7 | 0.5687 | 0.4729 | 0.6926 | 0 | 0 | 1.01973 | 0.227701 | 0.165281 | 19.1771 |

## Delta Table

| experiment | val_acc_delta | val_loss_delta | curvature_min_fraction_delta | curvature_collapse_penalty_delta | pred_abs_c_mean_delta | corr_complexity_abs_c_val_delta | corr_complexity_jac_val_delta | jac_frob_mean_delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 0.0758 | 0.0024 | 0 | 0 | -0.392408 | -0.000707984 | -0.184542 | -103214 |
| bert_no_jacobian_reg | 0.0939 | 0.0647 | 0 | 0 | -0.105315 | -0.0610728 | 0.513182 | -5.46448 |
| bert_no_curvature_aux | 0.0542 | -0.0001 | 0 | 0 | 0.230678 | 0.0340955 | -0.17705 | -268238 |
| bert_no_aux_all | -0.0542 | -0.0004 | 0 | 0 | 0.0998105 | 0.404738 | 0.415162 | 9.69066 |

## Reading Guide

- `val_acc_delta > 0`이면 epoch이 늘면서 검증 정확도가 오른 것이다.
- `val_loss_delta < 0`이면 검증 손실이 줄어든 것이다.
- `curvature_min_fraction`이 1에 가까워지면 곡률 collapse가 심해진 것이다.
- `corr_complexity_abs_c_val`이 양수로 커지면 복잡도와 곡률 연결이 강해진 것이다.
- `jac_frob_mean`이 지나치게 커지면 좌표 변환 반응이 폭주할 수 있다.

