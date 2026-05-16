# Codex Next Action Prompt

- cycle: `268`
- last status: `dry_run_planned`
- last commit: `6f225486b97590df8fcde8ed40eaf2d3da64d83b`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
