# Codex Next Action Prompt

- cycle: `177`
- last status: `dry_run_planned`
- last commit: `74a2fcc271a0b5481d558652804c6f00a5e37aa6`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
