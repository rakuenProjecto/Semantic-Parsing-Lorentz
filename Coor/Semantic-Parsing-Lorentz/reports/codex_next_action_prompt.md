# Codex Next Action Prompt

- cycle: `93`
- last status: `dry_run_planned`
- last commit: `2a2e66113fcaa4c618575ae3c8fc27e7e4a5e26a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
