# Codex Next Action Prompt

- cycle: `112`
- last status: `dry_run_planned`
- last commit: `8998ff652f78aed512badbf7ff4456e26950e66a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
