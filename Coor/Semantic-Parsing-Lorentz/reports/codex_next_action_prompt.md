# Codex Next Action Prompt

- cycle: `69`
- last status: `dry_run_planned`
- last commit: `840b048a07e3d2633d4ff9e260eab988713083fc`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
