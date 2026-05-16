# Codex Next Action Prompt

- cycle: `178`
- last status: `dry_run_planned`
- last commit: `b703ef75700b23f0976818b80962fa838b1ba608`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
