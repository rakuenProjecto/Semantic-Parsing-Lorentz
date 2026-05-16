# Codex Next Action Prompt

- cycle: `185`
- last status: `dry_run_planned`
- last commit: `d74e297904c697b67e5eeff89d5ea3aa6801832b`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
