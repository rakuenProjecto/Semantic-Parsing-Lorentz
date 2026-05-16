# Codex Next Action Prompt

- cycle: `233`
- last status: `dry_run_planned`
- last commit: `bafb1a1595b7179b1a259a6afa91ef3408a461ca`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
