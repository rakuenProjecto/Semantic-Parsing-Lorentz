# Codex Next Action Prompt

- cycle: `76`
- last status: `dry_run_planned`
- last commit: `b2fa880ad046cf45b0f82d29312f9eac9e69b478`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
