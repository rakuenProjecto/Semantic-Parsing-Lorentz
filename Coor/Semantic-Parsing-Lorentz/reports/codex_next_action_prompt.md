# Codex Next Action Prompt

- cycle: `64`
- last status: `dry_run_planned`
- last commit: `640392095bddb978a56b63817e5585041ad14306`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
