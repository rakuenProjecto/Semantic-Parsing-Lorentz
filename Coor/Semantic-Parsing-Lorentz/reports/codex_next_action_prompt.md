# Codex Next Action Prompt

- cycle: `156`
- last status: `dry_run_planned`
- last commit: `647502dc12a412345e66fd8c1cffe5d3ca614e84`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
