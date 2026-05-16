# Codex Next Action Prompt

- cycle: `24`
- last status: `dry_run_planned`
- last commit: `523daf678c9550bda40dde9807071269d0ea5909`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
