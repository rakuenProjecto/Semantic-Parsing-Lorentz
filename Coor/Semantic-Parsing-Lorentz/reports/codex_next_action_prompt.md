# Codex Next Action Prompt

- cycle: `17`
- last status: `dry_run_planned`
- last commit: `3465ae300b52f7a61fbc25cf974c4bf2f998d28c`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
