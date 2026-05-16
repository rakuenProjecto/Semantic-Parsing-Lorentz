# Codex Next Action Prompt

- cycle: `190`
- last status: `dry_run_planned`
- last commit: `fe33a545fd8d1bf0bfa7e0bc251a15ae0a8934b1`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
