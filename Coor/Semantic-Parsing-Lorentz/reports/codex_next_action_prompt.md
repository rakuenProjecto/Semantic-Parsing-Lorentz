# Codex Next Action Prompt

- cycle: `234`
- last status: `dry_run_planned`
- last commit: `c607ec5983d39ce4f06a6547c65fb5e3d8c93fd5`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
