# Codex Next Action Prompt

- cycle: `31`
- last status: `dry_run_planned`
- last commit: `250fe483ef66ba06d77a3c34175d23f2da303e48`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
