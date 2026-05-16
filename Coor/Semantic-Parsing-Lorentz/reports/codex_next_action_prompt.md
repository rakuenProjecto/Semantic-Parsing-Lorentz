# Codex Next Action Prompt

- cycle: `252`
- last status: `dry_run_planned`
- last commit: `e77c5c3eeab0190988e5b6279845ab8c8c7549d5`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
