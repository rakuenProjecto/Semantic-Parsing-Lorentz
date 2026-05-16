# Codex Next Action Prompt

- cycle: `20`
- last status: `dry_run_planned`
- last commit: `1d21285d13daad82355ff7c4ff2d420f82b572d8`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
