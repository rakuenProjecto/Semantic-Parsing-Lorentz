# Codex Next Action Prompt

- cycle: `202`
- last status: `dry_run_planned`
- last commit: `b5c34038c8a4693787f7f7e70e5caebbb7dbeb9d`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
