# Codex Next Action Prompt

- cycle: `28`
- last status: `dry_run_planned`
- last commit: `9a550770cd257d95f738501f559dfa92cd3a8b99`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
