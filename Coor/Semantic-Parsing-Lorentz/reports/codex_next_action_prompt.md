# Codex Next Action Prompt

- cycle: `245`
- last status: `dry_run_planned`
- last commit: `765fa4b1325774475744559ec32424c1ec09c37b`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
