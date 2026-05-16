# Codex Next Action Prompt

- cycle: `11`
- last status: `dry_run_planned`
- last commit: `36d174e1f63ca90e19dc6781d2f3408377165a07`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
