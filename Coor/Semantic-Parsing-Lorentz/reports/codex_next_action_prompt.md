# Codex Next Action Prompt

- cycle: `6`
- last status: `dry_run_planned`
- last commit: `acd64ee7ccc26db593bfbf98fc46cbc1dfff3ddc`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
