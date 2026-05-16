# Codex Next Action Prompt

- cycle: `55`
- last status: `dry_run_planned`
- last commit: `2f2e57d06cebab58f2ac4ccfc390c8ff84b7d42a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
