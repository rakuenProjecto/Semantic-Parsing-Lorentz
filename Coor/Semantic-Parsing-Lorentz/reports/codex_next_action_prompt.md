# Codex Next Action Prompt

- cycle: `259`
- last status: `dry_run_planned`
- last commit: `9c6b30381f1c30c1620efc363edc7df7ebd33b10`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
