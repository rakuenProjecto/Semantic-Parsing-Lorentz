# Codex Next Action Prompt

- cycle: `96`
- last status: `dry_run_planned`
- last commit: `76fb6b723310a9722995387ad8bdffb7af069997`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
