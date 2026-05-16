# Codex Next Action Prompt

- cycle: `129`
- last status: `dry_run_planned`
- last commit: `a563f9424764e0064b89fb9a7cb6a0d5e4b833e4`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
