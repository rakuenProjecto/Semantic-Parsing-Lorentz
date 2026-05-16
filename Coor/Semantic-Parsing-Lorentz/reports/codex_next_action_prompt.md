# Codex Next Action Prompt

- cycle: `204`
- last status: `dry_run_planned`
- last commit: `465b557b99df93e61e1c940388b4f3e1e4a606b1`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
