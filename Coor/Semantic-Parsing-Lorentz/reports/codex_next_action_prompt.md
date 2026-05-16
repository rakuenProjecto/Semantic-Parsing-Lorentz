# Codex Next Action Prompt

- cycle: `81`
- last status: `dry_run_planned`
- last commit: `d2be838743443290f3497a9c8a25092186d404a2`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
