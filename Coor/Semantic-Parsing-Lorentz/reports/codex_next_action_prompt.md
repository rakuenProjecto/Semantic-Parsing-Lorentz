# Codex Next Action Prompt

- cycle: `89`
- last status: `dry_run_planned`
- last commit: `de7b48b1ba7f62a76018e0bf90bc271492363dd1`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
