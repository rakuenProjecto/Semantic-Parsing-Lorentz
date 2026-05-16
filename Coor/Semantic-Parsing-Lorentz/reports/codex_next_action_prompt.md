# Codex Next Action Prompt

- cycle: `88`
- last status: `dry_run_planned`
- last commit: `1476668180d750178ba16a1288c4a8cd0e61beb1`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
