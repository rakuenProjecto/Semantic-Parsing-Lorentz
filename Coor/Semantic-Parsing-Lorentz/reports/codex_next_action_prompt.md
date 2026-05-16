# Codex Next Action Prompt

- cycle: `131`
- last status: `dry_run_planned`
- last commit: `a1729aeaf023754898c94239962d23a4676bfc83`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
