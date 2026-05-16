# Codex Next Action Prompt

- cycle: `165`
- last status: `dry_run_planned`
- last commit: `bf456896f9d2fd2a8eeb388f728404f64583743e`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
