# Codex Next Action Prompt

- cycle: `239`
- last status: `dry_run_planned`
- last commit: `aac11dd4e1397626c8a2f7c5de1cb1cce4b2b1ce`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
