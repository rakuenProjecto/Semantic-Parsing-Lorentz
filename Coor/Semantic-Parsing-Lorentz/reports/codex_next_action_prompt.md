# Codex Next Action Prompt

- cycle: `12`
- last status: `dry_run_planned`
- last commit: `98e22900c6aa7b3f4cf2c1d658a00ac9590a200e`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
