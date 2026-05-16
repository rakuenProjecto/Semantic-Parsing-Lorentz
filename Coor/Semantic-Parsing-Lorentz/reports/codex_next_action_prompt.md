# Codex Next Action Prompt

- cycle: `255`
- last status: `dry_run_planned`
- last commit: `23c33ffcc6bd1b3aac85c07e380195120f436391`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
