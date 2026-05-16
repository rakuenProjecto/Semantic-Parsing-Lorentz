# Codex Next Action Prompt

- cycle: `62`
- last status: `dry_run_planned`
- last commit: `3ae33c622c1c914db356e719e5fddad04e96e3a5`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
