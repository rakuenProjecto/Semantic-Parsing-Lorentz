# Codex Next Action Prompt

- cycle: `65`
- last status: `dry_run_planned`
- last commit: `ed135c16084b146a20391e721dc26a2e934d2f3a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
