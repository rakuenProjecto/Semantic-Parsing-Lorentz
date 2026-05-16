# Codex Next Action Prompt

- cycle: `205`
- last status: `dry_run_planned`
- last commit: `f8b3f27fcc00838e8585dbcf8dd056402b4cb3b7`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
