# Codex Next Action Prompt

- cycle: `231`
- last status: `dry_run_planned`
- last commit: `c0b76d7d6a676ca4f7dffe9dd8d1e68ed74ccc67`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
