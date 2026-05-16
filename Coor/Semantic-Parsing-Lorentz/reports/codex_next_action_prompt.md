# Codex Next Action Prompt

- cycle: `115`
- last status: `dry_run_planned`
- last commit: `3a8c6edfffe0e534a6ff14fe0af7937bfe39dccc`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
