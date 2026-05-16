# Codex Next Action Prompt

- cycle: `199`
- last status: `dry_run_planned`
- last commit: `9b1a5c42da7c882d1f7a1448897729be139a8dda`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
