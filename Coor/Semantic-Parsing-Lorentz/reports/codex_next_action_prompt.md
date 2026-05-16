# Codex Next Action Prompt

- cycle: `37`
- last status: `dry_run_planned`
- last commit: `581a19def3d406ebb8f692ef5205c8f4b4f0622d`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
