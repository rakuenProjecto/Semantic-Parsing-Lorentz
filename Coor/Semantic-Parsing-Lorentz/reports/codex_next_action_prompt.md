# Codex Next Action Prompt

- cycle: `27`
- last status: `dry_run_planned`
- last commit: `fcc2aeef65b5dfa09bfe5ed4513f6fd2dbf4680f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
