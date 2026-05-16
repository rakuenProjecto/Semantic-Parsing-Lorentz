# Codex Next Action Prompt

- cycle: `61`
- last status: `dry_run_planned`
- last commit: `6db1cbf142f6642567ee5432f9c0dc31327b7253`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
