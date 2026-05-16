# Codex Next Action Prompt

- cycle: `22`
- last status: `dry_run_planned`
- last commit: `abf18362b6d2b5d77179048a327f8c8e6a624562`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
