# Codex Next Action Prompt

- cycle: `212`
- last status: `dry_run_planned`
- last commit: `f41e112ee91917dea6f7dd18f35710963a97d09d`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
