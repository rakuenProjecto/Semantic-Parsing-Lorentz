# Codex Next Action Prompt

- cycle: `210`
- last status: `dry_run_planned`
- last commit: `ed8da815662c3c15c5e92e4044bedd184fcb7c19`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
