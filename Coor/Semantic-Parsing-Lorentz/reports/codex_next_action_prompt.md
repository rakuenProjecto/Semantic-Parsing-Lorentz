# Codex Next Action Prompt

- cycle: `7`
- last status: `dry_run_planned`
- last commit: `2b4cebcde1ce9f26a578954cbf919921093f4cc5`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
