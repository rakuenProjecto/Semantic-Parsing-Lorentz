# Codex Next Action Prompt

- cycle: `68`
- last status: `dry_run_planned`
- last commit: `61698e66e287939942c39738b2347b54c1cb7f1a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
