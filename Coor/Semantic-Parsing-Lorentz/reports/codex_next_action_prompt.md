# Codex Next Action Prompt

- cycle: `228`
- last status: `dry_run_planned`
- last commit: `8a9f5043b215331ce1f43e54f8d99d5cb69d8764`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
