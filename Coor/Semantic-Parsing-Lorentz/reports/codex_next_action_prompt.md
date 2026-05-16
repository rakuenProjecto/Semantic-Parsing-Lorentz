# Codex Next Action Prompt

- cycle: `19`
- last status: `dry_run_planned`
- last commit: `a50f3dc46a79b84f59327cfde7a2e3e33ee640e5`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
