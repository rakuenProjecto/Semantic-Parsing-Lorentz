# Codex Next Action Prompt

- cycle: `56`
- last status: `dry_run_planned`
- last commit: `cf9ed23dc33e865a1d5344c530ed1edfd77057f6`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
