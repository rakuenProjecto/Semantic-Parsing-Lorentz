# Codex Next Action Prompt

- cycle: `194`
- last status: `dry_run_planned`
- last commit: `709fe60970b8eab1df8b5646deaa725617a7b2bb`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
