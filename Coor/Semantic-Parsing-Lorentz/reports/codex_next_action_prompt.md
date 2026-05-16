# Codex Next Action Prompt

- cycle: `86`
- last status: `dry_run_planned`
- last commit: `99d63c4441e4dee67d7a7c1d76749793591219fd`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
