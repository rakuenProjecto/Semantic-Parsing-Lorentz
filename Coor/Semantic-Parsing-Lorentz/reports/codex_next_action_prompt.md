# Codex Next Action Prompt

- cycle: `120`
- last status: `dry_run_planned`
- last commit: `ab087e6e96488ae3a5eecd788c3167c439b8968d`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
