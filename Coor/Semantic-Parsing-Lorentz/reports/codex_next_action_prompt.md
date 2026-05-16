# Codex Next Action Prompt

- cycle: `141`
- last status: `dry_run_planned`
- last commit: `d345d06d6180be41e80ece91f37dc1d7a9e3310f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
