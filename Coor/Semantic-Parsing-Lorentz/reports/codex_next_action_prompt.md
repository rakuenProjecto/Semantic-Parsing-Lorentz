# Codex Next Action Prompt

- cycle: `240`
- last status: `dry_run_planned`
- last commit: `cd29332b234c54e41faac3b6fc5db342395bb939`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
