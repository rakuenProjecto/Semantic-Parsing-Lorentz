# Codex Next Action Prompt

- cycle: `269`
- last status: `dry_run_planned`
- last commit: `4a05bec37af52e4a5b64415d1d49db28d5030490`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
