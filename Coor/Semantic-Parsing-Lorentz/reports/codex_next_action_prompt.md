# Codex Next Action Prompt

- cycle: `66`
- last status: `dry_run_planned`
- last commit: `17bc2b41711070f51dfebf98ecaf8ef2e91d661d`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
