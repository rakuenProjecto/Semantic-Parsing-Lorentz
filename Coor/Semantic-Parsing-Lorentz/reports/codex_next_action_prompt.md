# Codex Next Action Prompt

- cycle: `60`
- last status: `dry_run_planned`
- last commit: `8121b43693c9b130a0ccd45652be198089280763`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
