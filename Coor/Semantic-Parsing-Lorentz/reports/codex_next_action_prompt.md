# Codex Next Action Prompt

- cycle: `109`
- last status: `dry_run_planned`
- last commit: `6c983a762f65bd96689170bd2e7253ebeed0d2fb`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
