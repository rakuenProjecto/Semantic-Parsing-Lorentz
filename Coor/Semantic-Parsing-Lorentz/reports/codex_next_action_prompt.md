# Codex Next Action Prompt

- cycle: `244`
- last status: `dry_run_planned`
- last commit: `5aaacf968a35cff0e2d80c7e4fa337b343386a69`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
