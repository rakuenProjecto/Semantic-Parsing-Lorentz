# Codex Next Action Prompt

- cycle: `253`
- last status: `dry_run_planned`
- last commit: `c6a44fa3bb37d21ecf64ee1af534d82e706ea13a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
