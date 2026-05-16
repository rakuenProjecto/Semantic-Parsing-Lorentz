# Codex Next Action Prompt

- cycle: `272`
- last status: `dry_run_planned`
- last commit: `66b14d4c86ddc38fff128701544132d80b326b6f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
