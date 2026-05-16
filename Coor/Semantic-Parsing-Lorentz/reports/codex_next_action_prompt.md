# Codex Next Action Prompt

- cycle: `29`
- last status: `dry_run_planned`
- last commit: `b23dfe5550d1d7f08d2a5d30509fd03909510f31`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
