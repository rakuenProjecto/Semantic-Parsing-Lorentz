# Codex Next Action Prompt

- cycle: `87`
- last status: `dry_run_planned`
- last commit: `d6342334faa4feb3cc9d8d915de8fe1cb688b2f3`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
