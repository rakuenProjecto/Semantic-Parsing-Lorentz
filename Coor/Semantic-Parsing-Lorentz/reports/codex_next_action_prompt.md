# Codex Next Action Prompt

- cycle: `74`
- last status: `dry_run_planned`
- last commit: `10c29db154a5c66fa2df8618246c076704ecb8cf`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
