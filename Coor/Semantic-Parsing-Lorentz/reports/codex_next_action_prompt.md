# Codex Next Action Prompt

- cycle: `150`
- last status: `dry_run_planned`
- last commit: `1104fb26549ba89e3af8cb0bf7f34eb5ccbb7720`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
