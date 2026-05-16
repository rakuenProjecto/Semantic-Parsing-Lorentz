# Codex Next Action Prompt

- cycle: `9`
- last status: `dry_run_planned`
- last commit: `ca1763a32e14a777057a049b785e28af9691993b`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
