# Codex Next Action Prompt

- cycle: `50`
- last status: `dry_run_planned`
- last commit: `fb495a0422ee7ade162540aa14247bb2cf71a3fe`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
