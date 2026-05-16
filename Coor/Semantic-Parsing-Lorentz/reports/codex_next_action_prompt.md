# Codex Next Action Prompt

- cycle: `36`
- last status: `dry_run_planned`
- last commit: `8e1e7a02cf27386154328c0159c2e0d0b68b1f90`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
