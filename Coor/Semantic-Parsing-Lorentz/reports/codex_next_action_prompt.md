# Codex Next Action Prompt

- cycle: `130`
- last status: `dry_run_planned`
- last commit: `ee94a0492545115b5704647090e2c7b0245294c7`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
