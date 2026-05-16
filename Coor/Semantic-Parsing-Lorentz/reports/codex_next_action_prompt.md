# Codex Next Action Prompt

- cycle: `71`
- last status: `dry_run_planned`
- last commit: `cc36a0ab894ee55544ef644290f5bb31398313ee`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
