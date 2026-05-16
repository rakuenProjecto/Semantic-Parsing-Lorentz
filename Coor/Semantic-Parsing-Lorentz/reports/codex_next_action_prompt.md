# Codex Next Action Prompt

- cycle: `203`
- last status: `dry_run_planned`
- last commit: `d0845570edd5dd575e341b35955fee7dbc6f826c`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
