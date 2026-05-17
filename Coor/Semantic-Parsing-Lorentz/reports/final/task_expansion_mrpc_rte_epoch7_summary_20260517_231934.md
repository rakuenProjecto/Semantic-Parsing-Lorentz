# Task Expansion Summary: MRPC + RTE

## Epoch 7 Summary Table

| task | experiment | val_acc | val_loss | val_majority_acc | val_acc_minus_majority | curvature_min_fraction | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mrpc | main | 0.8137 | 0.4822 |  |  | 0 | 1.22229 | 0.196877 | -0.111866 | 1.62048 |
| mrpc | bert_no_jacobian_reg | 0.7966 | 0.5041 |  |  | 0 | 0.99597 | 0.164673 | 0.220742 | 3620.4 |
| mrpc | bert_no_curvature_aux | 0.6838 | 0.625 |  |  | 0 | 0.0914412 | -0.290161 | -0.293492 | 6.16236 |
| mrpc | bert_no_aux_all | 0.6838 | 0.6773 |  |  | 0 | 0.37865 | -0.311822 | -0.161933 | 1.60517 |
| rte | main | 0.5632 | 0.6947 | 0.527076 | 0.0361242 | 0 | 1.30934 | 0.826813 | 0.610532 | 1499.56 |
| rte | bert_no_jacobian_reg | 0.6065 | 0.7572 | 0.527076 | 0.0794242 | 0 | 1.1785 | 0.724107 | 0.171118 | 7.44829 |
| rte | bert_no_curvature_aux | 0.5271 | 0.6931 | 0.527076 | 2.41877e-05 | 0 | 1.93933 | 0.798007 | -0.449529 | 522061 |
| rte | bert_no_aux_all | 0.4729 | 0.6926 | 0.527076 | -0.0541758 | 0 | 1.01973 | 0.227701 | 0.165281 | 19.1771 |

## Reading

- `val_acc_minus_majority`가 클수록 다수 라벨만 찍는 기준선보다 실제로 더 학습한 것이다.
- `curvature_min_fraction`이 1에 가까우면 곡률이 최소값으로 몰린 것이다.
- `jac_frob_mean`이 지나치게 크면 좌표 변환 반응이 폭주한 것으로 볼 수 있다.
- MRPC는 main이 다수 라벨 기준선을 크게 넘는지, RTE는 작은 validation 크기 때문에 seed 반복이 필요한지를 함께 본다.

