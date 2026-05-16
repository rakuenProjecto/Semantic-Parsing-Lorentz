# Codex Next Action Prompt

- cycle: `166`
- last status: `dry_run_planned`
- last commit: `961cebef45ed49af6b920c0e757d750e9ca108c2`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
