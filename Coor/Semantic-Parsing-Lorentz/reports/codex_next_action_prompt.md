# Codex Next Action Prompt

- cycle: `241`
- last status: `dry_run_planned`
- last commit: `d1ab7966986a1cefbd24166cf032f77af38a89fa`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
