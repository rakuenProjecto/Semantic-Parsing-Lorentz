# Codex Next Action Prompt

- cycle: `134`
- last status: `dry_run_planned`
- last commit: `a37f8ba5c7d984a0c58d061d1dbaa0fd563788ab`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
