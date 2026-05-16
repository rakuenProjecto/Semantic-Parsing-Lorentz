# Codex Next Action Prompt

- cycle: `171`
- last status: `dry_run_planned`
- last commit: `f9f8314a5f75ec6127f6565969db69995bad8ee7`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
