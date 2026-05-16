# Codex Next Action Prompt

- cycle: `151`
- last status: `dry_run_planned`
- last commit: `3df9b3ee572565d7ec9bae6089c5e48112525f4f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
