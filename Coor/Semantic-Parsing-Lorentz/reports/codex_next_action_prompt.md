# Codex Next Action Prompt

- cycle: `132`
- last status: `dry_run_planned`
- last commit: `7d1540333883fe4d783da951ff3e5f09369c8b18`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
