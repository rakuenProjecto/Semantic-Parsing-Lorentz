# Codex Next Action Prompt

- cycle: `208`
- last status: `dry_run_planned`
- last commit: `94e86ce3485c1cd4cf60ecc7ab6d01bbae30d00c`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
