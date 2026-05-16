# Codex Next Action Prompt

- cycle: `195`
- last status: `dry_run_planned`
- last commit: `24de40c00fedc6f3ae8097789c6126319b181614`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
