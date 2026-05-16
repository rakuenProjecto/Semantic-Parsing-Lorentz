# Codex Next Action Prompt

- cycle: `270`
- last status: `dry_run_planned`
- last commit: `4b1a6dfdab6b1f00a13335bdbc1d793852ac1ee2`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
