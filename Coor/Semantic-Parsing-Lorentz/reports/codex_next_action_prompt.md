# Codex Next Action Prompt

- cycle: `77`
- last status: `dry_run_planned`
- last commit: `fa0d6915ad72230dc4f8d4dbe33dceca44e23dd3`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
