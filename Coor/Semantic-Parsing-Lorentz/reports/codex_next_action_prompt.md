# Codex Next Action Prompt

- cycle: `235`
- last status: `dry_run_planned`
- last commit: `9f7eee19c8e4b7d0759815932d998efd3ee372e2`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
