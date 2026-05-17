# A10 Isolated SST-2 Ablation Comparison

- run_tag: `fresh_a10_isolated_n054.hpc_20260517_164753`
- output_root: `outputs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753`
- log_root: `logs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753`
- report_root: `reports_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753`

## Summary Table

| experiment | train_acc | val_acc | val_loss | curvature_min_fraction | curvature_collapse_penalty | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 0.9606 | 0.8979 | 0.4428 | 0 | 0 | 0.815642 | 0.857379 | 0.46973 | 1.29303 |
| bert_no_jacobian_reg | 0.969 | 0.9197 | 0.3368 | 0 | 0 | 0.842807 | 0.901586 | 0.23104 | 1.93221 |
| bert_no_curvature_aux | 0.9676 | 0.9083 | 0.4299 | 1 | 0.999791 | 0.0500209 | -0.0475253 | -0.2018 | 0.636095 |
| bert_no_aux_all | 0.9588 | 0.9014 | 0.3983 | 1 | 0.999214 | 0.0500786 | 0.0254212 | -0.0233847 | 0.890086 |

## Interpretation

- main은 val_acc=0.8979, curvature_min_fraction=0, corr_complexity_abs_c_val=0.857379를 보인다.
- bert_no_jacobian_reg는 val_acc=0.9197, jac_frob_mean=1.93221이다. main보다 jac_frob_mean이 커지면 Jacobian regularization이 좌표 변환 반응을 안정화하는 역할로 해석할 수 있다.
- bert_no_curvature_aux는 val_acc=0.9083, corr_complexity_abs_c_val=-0.0475253이다. 성능이나 복잡도-곡률 상관이 크게 떨어지면 곡률 보조 손실의 필요성을 보여준다.
- bert_no_aux_all은 val_acc=0.9014, curvature_min_fraction=1, pred_abs_c_mean=0.0500786이다. 정확도가 유지되더라도 curvature_min_fraction이 1에 가까우면 곡률 표현은 collapse된 것이다.

핵심은 accuracy만 보지 않고, curvature_min_fraction, corr_complexity_abs_c_val, corr_complexity_jac_val, jac_frob_mean을 함께 봐야 한다는 점이다.

## Paths

### main
- output_dir: `outputs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/main`
- log_path: `logs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/main.log`
- diagnostics_path: `outputs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/main/diagnostics_epoch_3.json`
- summary_path: `reports_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/main/latest_experiment_summary.md`

### bert_no_jacobian_reg
- output_dir: `outputs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_jacobian_reg`
- log_path: `logs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_jacobian_reg.log`
- diagnostics_path: `outputs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_jacobian_reg/diagnostics_epoch_3.json`
- summary_path: `reports_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_jacobian_reg/latest_experiment_summary.md`

### bert_no_curvature_aux
- output_dir: `outputs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_curvature_aux`
- log_path: `logs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_curvature_aux.log`
- diagnostics_path: `outputs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_curvature_aux/diagnostics_epoch_3.json`
- summary_path: `reports_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_curvature_aux/latest_experiment_summary.md`

### bert_no_aux_all
- output_dir: `outputs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_aux_all`
- log_path: `logs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_aux_all.log`
- diagnostics_path: `outputs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_aux_all/diagnostics_epoch_3.json`
- summary_path: `reports_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753/bert_no_aux_all/latest_experiment_summary.md`

