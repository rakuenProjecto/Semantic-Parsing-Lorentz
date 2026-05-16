# Codex Next Action Prompt

- cycle: `35`
- last status: `dry_run_planned`
- last commit: `f0b8630ccde97a19beed2f8fc622545547c2ea7e`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
