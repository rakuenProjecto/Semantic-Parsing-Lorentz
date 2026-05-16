# Codex Next Action Prompt

- cycle: `160`
- last status: `dry_run_planned`
- last commit: `2ea94d3d4432275378b8e76aaad67ecad1d858b6`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
