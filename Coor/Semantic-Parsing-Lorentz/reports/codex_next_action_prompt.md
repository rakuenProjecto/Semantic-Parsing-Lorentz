# Codex Next Action Prompt

- cycle: `263`
- last status: `dry_run_planned`
- last commit: `e79984c8b4365430285029fe0b7930e9a000fbfe`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
