# Codex Next Action Prompt

- cycle: `111`
- last status: `dry_run_planned`
- last commit: `0f06a1600921fa9cf3a7b8cccba75f58dd66c033`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
