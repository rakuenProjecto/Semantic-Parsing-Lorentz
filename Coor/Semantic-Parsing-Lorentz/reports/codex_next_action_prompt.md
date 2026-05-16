# Codex Next Action Prompt

- cycle: `70`
- last status: `dry_run_planned`
- last commit: `818cdf0724eb425b791c3e8c5fd3876c1c102c29`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
