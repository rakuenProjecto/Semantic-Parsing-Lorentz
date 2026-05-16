# Codex Next Action Prompt

- cycle: `143`
- last status: `dry_run_planned`
- last commit: `3fe7a10e178832f88151353a8f427fbe5acd3d54`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
