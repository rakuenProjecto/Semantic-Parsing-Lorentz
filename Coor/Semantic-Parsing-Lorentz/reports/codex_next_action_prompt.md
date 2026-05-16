# Codex Next Action Prompt

- cycle: `52`
- last status: `dry_run_planned`
- last commit: `d657b181df5289497460f95a8c200f0b178184f6`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
