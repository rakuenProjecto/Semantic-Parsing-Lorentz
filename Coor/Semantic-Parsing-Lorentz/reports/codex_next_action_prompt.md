# Codex Next Action Prompt

- cycle: `223`
- last status: `dry_run_planned`
- last commit: `27b59f3dde128a8f3fdc2b962b0fe1e4b8282a18`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
