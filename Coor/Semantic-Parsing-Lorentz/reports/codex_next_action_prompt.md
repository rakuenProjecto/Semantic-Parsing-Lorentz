# Codex Next Action Prompt

- cycle: `127`
- last status: `dry_run_planned`
- last commit: `319d0d998d8b492717780aa08e5997c8ac1c4ec0`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
