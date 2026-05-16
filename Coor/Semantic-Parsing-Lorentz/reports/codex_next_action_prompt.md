# Codex Next Action Prompt

- cycle: `97`
- last status: `dry_run_planned`
- last commit: `3cb13460ba41fbcf85e0d75198bfbb83639abc7f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
