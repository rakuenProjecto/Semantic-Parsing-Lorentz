# Codex Next Action Prompt

- cycle: `119`
- last status: `dry_run_planned`
- last commit: `48d92fc537cdf0bfad3232c737862206521825fc`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
