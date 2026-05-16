# Codex Next Action Prompt

- cycle: `90`
- last status: `dry_run_planned`
- last commit: `a50a2ea81d454011903d092911d331c357242d0f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
