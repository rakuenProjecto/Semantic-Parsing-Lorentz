# MRPC Epoch 3 → Epoch 7 Trend Comparison

- task: `mrpc`
- run_tag: `glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443`
- output_root: `outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443`
- log_root: `logs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443`

## Epoch Final Values

| experiment | epoch | train_acc | val_acc | val_loss | curvature_min_fraction | curvature_collapse_penalty | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 3 | 0.7017 | 0.7623 | 0.5378 | 0 | 0 | 0.981264 | -0.186066 | -0.136237 | 2.51516 |
| main | 7 | 0.8539 | 0.8137 | 0.4822 | 0 | 0 | 1.22229 | 0.196877 | -0.111866 | 1.62048 |
| bert_no_jacobian_reg | 3 | 0.702 | 0.6765 | 0.6361 | 0 | 0 | 1.43096 | 0.304185 | 0.310577 | 86.7539 |
| bert_no_jacobian_reg | 7 | 0.907 | 0.7966 | 0.5041 | 0 | 0 | 0.99597 | 0.164673 | 0.220742 | 3620.4 |
| bert_no_curvature_aux | 3 | 0.6745 | 0.6838 | 0.626 | 0 | 0.292954 | 0.14175 | -0.567593 | 0.486695 | 2.11942 |
| bert_no_curvature_aux | 7 | 0.6745 | 0.6838 | 0.625 | 0 | 0.628522 | 0.0914412 | -0.290161 | -0.293492 | 6.16236 |
| bert_no_aux_all | 3 | 0.6745 | 0.6838 | 0.6859 | 0 | 0 | 0.378317 | -0.311676 | -0.162193 | 1.60702 |
| bert_no_aux_all | 7 | 0.6745 | 0.6838 | 0.6773 | 0 | 0 | 0.37865 | -0.311822 | -0.161933 | 1.60517 |

## Delta Table

| experiment | val_acc_delta | val_loss_delta | curvature_min_fraction_delta | curvature_collapse_penalty_delta | pred_abs_c_mean_delta | corr_complexity_abs_c_val_delta | corr_complexity_jac_val_delta | jac_frob_mean_delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 0.0514 | -0.0556 | 0 | 0 | 0.241025 | 0.382943 | 0.0243709 | -0.894675 |
| bert_no_jacobian_reg | 0.1201 | -0.132 | 0 | 0 | -0.434986 | -0.139512 | -0.089835 | 3533.64 |
| bert_no_curvature_aux | 0 | -0.001 | 0 | 0.335568 | -0.0503084 | 0.277431 | -0.780187 | 4.04294 |
| bert_no_aux_all | 0 | -0.0086 | 0 | 0 | 0.000333577 | -0.000146627 | 0.000259146 | -0.0018481 |

## Reading Guide

- `val_acc_delta > 0`이면 epoch이 늘면서 검증 정확도가 오른 것이다.
- `val_loss_delta < 0`이면 검증 손실이 줄어든 것이다.
- `curvature_min_fraction`이 1에 가까워지면 곡률 collapse가 심해진 것이다.
- `corr_complexity_abs_c_val`이 양수로 커지면 복잡도와 곡률 연결이 강해진 것이다.
- `jac_frob_mean`이 지나치게 커지면 좌표 변환 반응이 폭주할 수 있다.

