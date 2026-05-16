# Codex Next Action Prompt

- cycle: `230`
- last status: `dry_run_planned`
- last commit: `0134583b9a66f91eacc4111fce435887ea4eb276`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
