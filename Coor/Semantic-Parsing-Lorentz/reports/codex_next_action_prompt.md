# Codex Next Action Prompt

- cycle: `125`
- last status: `dry_run_planned`
- last commit: `cc0f07c5ddebb379a721668c343be939cb29b3f8`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
