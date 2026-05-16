# Codex Next Action Prompt

- cycle: `33`
- last status: `dry_run_planned`
- last commit: `d5f25586b4906f8eeb990f5fee1ed7cc7d1b8786`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
