# Codex Next Action Prompt

- cycle: `152`
- last status: `dry_run_planned`
- last commit: `5db6caf46df62f145cdf0fdebc5bb8b1333e96b8`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
