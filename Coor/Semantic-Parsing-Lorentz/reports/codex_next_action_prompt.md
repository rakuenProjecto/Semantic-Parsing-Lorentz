# Codex Next Action Prompt

- cycle: `182`
- last status: `dry_run_planned`
- last commit: `4b6aaa842bd02e6b8965adc2149ded2b63c40885`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
