# Codex Next Action Prompt

- cycle: `260`
- last status: `dry_run_planned`
- last commit: `f9420e6435f7e25e35c3a8fae375af26eb6cf0ca`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
