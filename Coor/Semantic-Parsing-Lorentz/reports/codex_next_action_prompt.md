# Codex Next Action Prompt

- cycle: `21`
- last status: `dry_run_planned`
- last commit: `21a13a6e6c211f23a51cbc4610cc85b33fc876b9`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
