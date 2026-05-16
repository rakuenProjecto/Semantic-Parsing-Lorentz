# Codex Next Action Prompt

- cycle: `34`
- last status: `dry_run_planned`
- last commit: `a46db963ce0d440ee4e4a5b9a918a8d22a31af93`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
