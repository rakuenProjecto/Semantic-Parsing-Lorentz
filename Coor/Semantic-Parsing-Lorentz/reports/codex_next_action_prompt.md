# Codex Next Action Prompt

- cycle: `179`
- last status: `dry_run_planned`
- last commit: `e69ef87cad90b4ee1b4744e74d8cbd1f93574171`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
