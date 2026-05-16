# Codex Next Action Prompt

- cycle: `236`
- last status: `dry_run_planned`
- last commit: `ac3e9eb396a5730ebdacac6bb7aa5d0858b555e9`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
