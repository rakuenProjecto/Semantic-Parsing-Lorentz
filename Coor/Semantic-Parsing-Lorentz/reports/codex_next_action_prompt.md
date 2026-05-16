# Codex Next Action Prompt

- cycle: `40`
- last status: `dry_run_planned`
- last commit: `a90759262094c7f94cc788d5427721484750c4a2`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
