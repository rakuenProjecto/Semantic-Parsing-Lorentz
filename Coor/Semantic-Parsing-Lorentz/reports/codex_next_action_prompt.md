# Codex Next Action Prompt

- cycle: `183`
- last status: `dry_run_planned`
- last commit: `7c00ca53b43a77caf7b0e6bd646a0b55050d63da`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
