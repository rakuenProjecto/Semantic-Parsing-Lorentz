# Codex Next Action Prompt

- cycle: `99`
- last status: `dry_run_planned`
- last commit: `15bb8aac1550c99703fd8db20cf36414ed887b0c`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
