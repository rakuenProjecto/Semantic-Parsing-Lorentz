# Codex Next Action Prompt

- cycle: `18`
- last status: `dry_run_planned`
- last commit: `ca8499d9c26dc732341f4baa426e7ee3ade9b9f8`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
