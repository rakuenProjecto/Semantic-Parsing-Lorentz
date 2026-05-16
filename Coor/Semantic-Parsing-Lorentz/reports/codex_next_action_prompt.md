# Codex Next Action Prompt

- cycle: `158`
- last status: `dry_run_planned`
- last commit: `15d9006478336d603a4a7206b3e43dee02c9a4eb`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
