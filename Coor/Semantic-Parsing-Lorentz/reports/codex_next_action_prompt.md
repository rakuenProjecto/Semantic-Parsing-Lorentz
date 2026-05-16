# Codex Next Action Prompt

- cycle: `45`
- last status: `dry_run_planned`
- last commit: `e1dd12374a03fb8b39b377c470f40966064c3255`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
