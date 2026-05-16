# Codex Next Action Prompt

- cycle: `262`
- last status: `dry_run_planned`
- last commit: `ffadff4d5533d7216e095381ce51b693372f1ec1`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
