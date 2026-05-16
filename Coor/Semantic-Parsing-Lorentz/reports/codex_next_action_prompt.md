# Codex Next Action Prompt

- cycle: `30`
- last status: `dry_run_planned`
- last commit: `4d85cf363556b09e43216fb8524c93c87f8cd8a2`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
