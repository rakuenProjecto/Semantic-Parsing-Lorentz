# Codex Next Action Prompt

- cycle: `139`
- last status: `dry_run_planned`
- last commit: `804d215d577c548d5a591706f6d8e3d1561cd821`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
