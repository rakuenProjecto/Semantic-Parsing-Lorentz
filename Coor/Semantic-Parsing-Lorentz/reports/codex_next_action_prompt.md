# Codex Next Action Prompt

- cycle: `237`
- last status: `dry_run_planned`
- last commit: `2956a87354595d7ec266485c750297432726a4c0`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
