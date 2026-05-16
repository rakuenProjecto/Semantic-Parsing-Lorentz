# Codex Next Action Prompt

- cycle: `23`
- last status: `dry_run_planned`
- last commit: `1a7dcbdd70394a645642ce66b1d7e30d207780f6`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
