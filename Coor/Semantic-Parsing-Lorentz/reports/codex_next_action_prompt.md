# Codex Next Action Prompt

- cycle: `53`
- last status: `dry_run_planned`
- last commit: `9c070bcb8db9a5bf7966b881ca661a65d704d8a0`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
