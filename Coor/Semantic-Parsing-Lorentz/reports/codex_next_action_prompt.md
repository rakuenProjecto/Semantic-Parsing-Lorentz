# Codex Next Action Prompt

- cycle: `95`
- last status: `dry_run_planned`
- last commit: `88e6a9d449f9324cdccee288d089af6b80324f2f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
