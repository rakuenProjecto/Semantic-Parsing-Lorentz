# Codex Next Action Prompt

- cycle: `144`
- last status: `dry_run_planned`
- last commit: `1158d4d4a14f545ef58523251c20e5392bfa3cc0`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
