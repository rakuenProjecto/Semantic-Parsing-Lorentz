# Codex Next Action Prompt

- cycle: `267`
- last status: `dry_run_planned`
- last commit: `bcf228dc1d12dff1a7d234ccf7206bce9ca651e5`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
