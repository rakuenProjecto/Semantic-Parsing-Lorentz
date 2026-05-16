# Codex Next Action Prompt

- cycle: `91`
- last status: `dry_run_planned`
- last commit: `50845719862f68bc0dd1b328129ec98af2d09525`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
