# RTE Ablation Numeric Analysis

## Source Files

- epoch7: `reports/final/rte_epoch7_task_ablation_comparison_glue_rte_e7_a10_isolated_n054.hpc_20260517_222501_20260517_231604.json`
- trend: `reports/final/rte_epoch3_to_epoch7_trend_glue_rte_e7_a10_isolated_n054.hpc_20260517_222501_20260517_231604.json`
- label_distribution: `reports/final/rte_label_distribution_20260517_231640.json`

## Epoch 7 Summary

| experiment | val_acc | val_loss | curvature_min_fraction | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| main | 0.5632 | 0.6947 | 0 | 1.30934 | 0.826813 | 0.610532 | 1499.56 |
| bert_no_jacobian_reg | 0.6065 | 0.7572 | 0 | 1.1785 | 0.724107 | 0.171118 | 7.44829 |
| bert_no_curvature_aux | 0.5271 | 0.6931 | 0 | 1.93933 | 0.798007 | -0.449529 | 522061 |
| bert_no_aux_all | 0.4729 | 0.6926 | 0 | 1.01973 | 0.227701 | 0.165281 | 19.1771 |

## Epoch 3 → 7 Change

| experiment | val_acc_delta | val_loss_delta | pred_abs_c_mean_delta | corr_abs_c_delta | corr_jac_delta | jac_frob_delta |
|---|---:|---:|---:|---:|---:|---:|
| main | 0.0758 | 0.0024 | -0.392408 | -0.000707984 | -0.184542 | -103214 |
| bert_no_jacobian_reg | 0.0939 | 0.0647 | -0.105315 | -0.0610728 | 0.513182 | -5.46448 |
| bert_no_curvature_aux | 0.0542 | -0.0001 | 0.230678 | 0.0340955 | -0.17705 | -268238 |
| bert_no_aux_all | -0.0542 | -0.0004 | 0.0998105 | 0.404738 | 0.415162 | 9.69066 |

## Interpretation

- RTE validation 다수 라벨 기준선은 `0.527076`이다.
- epoch 7에서 가장 높은 val_acc는 `bert_no_jacobian_reg`의 `0.6065`이다.
- epoch 7에서 가장 낮은 val_loss는 `bert_no_aux_all`의 `0.6926`이다.
- main은 다수 라벨 기준선보다 `0.0361242`만큼 높다.
- no_aux_all은 다수 라벨 기준선보다 `-0.0541758`만큼 높다.
- main과 no_aux_all의 val_acc 차이는 `0.0903`이다.
- no_jacobian_reg의 jac_frob_mean은 main 대비 약 `0.00496697`배이다.

### Reading

- main이 다수 라벨 기준선을 뚜렷하게 넘고, no_aux_all보다 높으면 구조 손실이 task 성능에도 기여한 근거가 된다.
- no_jacobian_reg의 jac_frob_mean이 main보다 크게 높으면 Jacobian 안정화 항이 좌표 변환 폭주를 막는 역할을 한 것으로 해석할 수 있다.
- no_curvature_aux 또는 no_aux_all의 pred_abs_c_mean이 작고 corr 값이 약하면 곡률 표현이 task 구조와 잘 연결되지 못한 것으로 볼 수 있다.
- RTE는 validation 샘플 수가 277개로 작기 때문에, 최종 주장에는 seed 반복이 필요하다.
