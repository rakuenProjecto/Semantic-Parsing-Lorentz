# Codex Next Action Prompt

- cycle: `8`
- last status: `dry_run_planned`
- last commit: `83929ffb2e96df2b00c8d49186eb37cd4ecaf639`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
