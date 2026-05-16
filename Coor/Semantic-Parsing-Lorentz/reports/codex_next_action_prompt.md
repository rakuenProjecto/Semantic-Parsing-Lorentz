# Codex Next Action Prompt

- cycle: `167`
- last status: `dry_run_planned`
- last commit: `8b3fe5b5273da91b460f43327d9d7311d79e6c10`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
