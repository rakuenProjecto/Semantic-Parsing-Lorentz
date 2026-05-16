# Codex Next Action Prompt

- cycle: `266`
- last status: `dry_run_planned`
- last commit: `c5715056a715a6d89c98d638e33214ecb8399c98`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
