# Codex Next Action Prompt

- cycle: `176`
- last status: `dry_run_planned`
- last commit: `dc09e7cf69effec4e1b5861908a6b36f05a2fff2`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
