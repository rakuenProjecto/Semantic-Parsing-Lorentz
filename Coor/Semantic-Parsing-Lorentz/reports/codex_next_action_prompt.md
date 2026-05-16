# Codex Next Action Prompt

- cycle: `142`
- last status: `dry_run_planned`
- last commit: `f26c3aaa8bd8a049ee31d86835cf70917e8d04d2`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
