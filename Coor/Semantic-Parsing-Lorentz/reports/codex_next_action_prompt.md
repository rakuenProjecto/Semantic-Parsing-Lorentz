# Codex Next Action Prompt

- cycle: `242`
- last status: `dry_run_planned`
- last commit: `76b76a144cce1919639764e2b45b0dc91d9ae5c6`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
