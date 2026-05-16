# Codex Next Action Prompt

- cycle: `173`
- last status: `dry_run_planned`
- last commit: `9f99fa0b352d0bc4faf60c2dd7c14173b5cf9105`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
