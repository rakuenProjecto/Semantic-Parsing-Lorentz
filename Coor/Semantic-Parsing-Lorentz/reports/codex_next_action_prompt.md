# Codex Next Action Prompt

- cycle: `189`
- last status: `dry_run_planned`
- last commit: `baf6f5dc6a274bf9e32e132c795d7fc8868d280a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
