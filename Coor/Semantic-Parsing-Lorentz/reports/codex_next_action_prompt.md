# Codex Next Action Prompt

- cycle: `218`
- last status: `dry_run_planned`
- last commit: `346a6cec81f531b12542bc2e0f6f1e7092644108`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
