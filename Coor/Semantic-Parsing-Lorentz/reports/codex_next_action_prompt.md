# Codex Next Action Prompt

- cycle: `161`
- last status: `dry_run_planned`
- last commit: `5c660dd04f6097ae4860a0686ecd4f2324e27d6e`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
