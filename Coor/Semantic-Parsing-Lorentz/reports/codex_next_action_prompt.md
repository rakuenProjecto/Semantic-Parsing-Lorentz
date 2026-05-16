# Codex Next Action Prompt

- cycle: `184`
- last status: `dry_run_planned`
- last commit: `4042d78fd9ca48a210cac37c94eb5c60b37a075d`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
