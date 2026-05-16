# Codex Next Action Prompt

- cycle: `229`
- last status: `dry_run_planned`
- last commit: `89ae3dc48990b06ff79428f96a8e9859dbe34fb1`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
