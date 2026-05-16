# Codex Next Action Prompt

- cycle: `243`
- last status: `dry_run_planned`
- last commit: `6c267e7c9cec56b1c9b5ad585f8504e59320816a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
