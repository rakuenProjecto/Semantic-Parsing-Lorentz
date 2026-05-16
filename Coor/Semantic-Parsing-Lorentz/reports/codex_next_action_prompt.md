# Codex Next Action Prompt

- cycle: `98`
- last status: `dry_run_planned`
- last commit: `c005ac41b71cca0079b501b008bbacbb9d8bd2dc`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
