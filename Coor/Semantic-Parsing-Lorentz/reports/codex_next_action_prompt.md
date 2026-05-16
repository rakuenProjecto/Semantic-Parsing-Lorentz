# Codex Next Action Prompt

- cycle: `39`
- last status: `dry_run_planned`
- last commit: `3aa0ab722dbb56404e001eb530885792845afc18`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
