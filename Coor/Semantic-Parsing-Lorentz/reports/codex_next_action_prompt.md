# Codex Next Action Prompt

- cycle: `84`
- last status: `dry_run_planned`
- last commit: `c304b978a2fb059a8ba8402df7e23848430643cd`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
