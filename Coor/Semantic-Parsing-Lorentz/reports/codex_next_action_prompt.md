# Codex Next Action Prompt

- cycle: `58`
- last status: `dry_run_planned`
- last commit: `052a6a5ecb05610677489b0d4062d2170b0b3e40`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
