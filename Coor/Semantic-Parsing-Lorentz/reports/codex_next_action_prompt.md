# Codex Next Action Prompt

- cycle: `110`
- last status: `dry_run_planned`
- last commit: `b6ec4e0ef9a88c058301014762f10f3837e8e899`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
