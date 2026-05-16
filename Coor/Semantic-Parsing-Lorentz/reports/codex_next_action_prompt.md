# Codex Next Action Prompt

- cycle: `133`
- last status: `dry_run_planned`
- last commit: `fe99b4e51a436caa2cd553b32e9cdc30a7203623`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
