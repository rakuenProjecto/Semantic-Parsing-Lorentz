# Codex Next Action Prompt

- cycle: `219`
- last status: `dry_run_planned`
- last commit: `74726d4d793ca1a9ea10e0545550717a8d8495d3`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
