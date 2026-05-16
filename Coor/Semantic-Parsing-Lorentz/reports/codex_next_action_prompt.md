# Codex Next Action Prompt

- cycle: `162`
- last status: `dry_run_planned`
- last commit: `8b22205ce37039471fed3e5906b10c67d30171a0`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
