# n001 SST-2 Ablation Comparison

## Summary Table

| experiment | train_acc | val_acc | val_loss | curvature_min_fraction | curvature_collapse_penalty | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 0.9627 | 0.9048 | 0.3924 | 0 | 0 | 0.781513 | 0.845326 | 0.516644 | 1.1126 |
| bert_no_jacobian_reg | 0.9635 | 0.9151 | 0.338 | 0 | 0 | 0.807944 | 0.84305 | 0.255858 | 2.09346 |
| bert_no_curvature_aux | 0.5579 | 0.5092 | 0.6976 | 0 | 0.00180248 | 0.241509 | -0.319896 | 0.0689311 | 0.264345 |
| bert_no_aux_all | 0.9666 | 0.9048 | 0.4447 | 1 | 0.997754 | 0.0502247 | -0.0347772 | -0.11727 | 0.951822 |

## Interpretation

- main은 val_acc=0.9048, curvature_min_fraction=0.0으로 정확도와 곡률 안정성을 동시에 유지했다.
- bert_no_jacobian_reg는 val_acc가 가장 높지만 jac_frob_mean이 크게 증가하여 Jacobian 안정화 항이 좌표 변환 반응을 억제하는 역할을 했을 가능성이 있다.
- bert_no_curvature_aux는 val_acc가 0.5092로 크게 하락하여 곡률 보조 손실 제거가 성능 안정성에 큰 악영향을 준 것으로 보인다.
- bert_no_aux_all은 val_acc는 main과 같지만 curvature_min_fraction=1.0으로 validation 전체가 최소 곡률 근처에 몰렸다. 즉, 정확도만 보면 괜찮아 보여도 곡률 표현은 완전히 collapse되었다.

## Paths

### main
- output_dir: `outputs/real_sst2_full_e10_20260517_014513`
- log_path: `logs/real_sst2_full_e10_20260517_014513.log`
- diagnostics_path: `outputs/real_sst2_full_e10_20260517_014513/diagnostics_epoch_3.json`

### bert_no_jacobian_reg
- output_dir: `outputs/ablations/bert_no_jacobian_reg_20260517_103225`
- log_path: `logs/ablations/bert_no_jacobian_reg_20260517_103225.log`
- diagnostics_path: `outputs/ablations/bert_no_jacobian_reg_20260517_103225/diagnostics_epoch_3.json`

### bert_no_curvature_aux
- output_dir: `outputs/ablations/bert_no_curvature_aux_20260517_103225`
- log_path: `logs/ablations/bert_no_curvature_aux_20260517_103225.log`
- diagnostics_path: `outputs/ablations/bert_no_curvature_aux_20260517_103225/diagnostics_epoch_3.json`

### bert_no_aux_all
- output_dir: `outputs/ablations/bert_no_aux_all_20260517_103225`
- log_path: `logs/ablations/bert_no_aux_all_20260517_103225.log`
- diagnostics_path: `outputs/ablations/bert_no_aux_all_20260517_103225/diagnostics_epoch_3.json`

