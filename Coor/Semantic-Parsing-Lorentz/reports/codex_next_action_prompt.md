# Codex Next Action Prompt

- cycle: `101`
- last status: `dry_run_planned`
- last commit: `40074eb8e50e335be6e3ba2c2e196fc79cbd77d8`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
