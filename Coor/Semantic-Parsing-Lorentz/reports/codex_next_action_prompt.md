# Codex Next Action Prompt

- cycle: `78`
- last status: `dry_run_planned`
- last commit: `f06cd7634086a3e74913d200d8058d3e60a88e7c`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
