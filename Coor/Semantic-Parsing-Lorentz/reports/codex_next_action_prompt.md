# Codex Next Action Prompt

- cycle: `103`
- last status: `dry_run_planned`
- last commit: `d790c7c6631e757dad3a09ca17c9f2c58eb3b877`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
