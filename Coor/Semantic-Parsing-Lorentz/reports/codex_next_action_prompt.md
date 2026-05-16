# Codex Next Action Prompt

- cycle: `257`
- last status: `dry_run_planned`
- last commit: `3b1f3dff2cd56b709b90193f57ec6b37f0941a0a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
