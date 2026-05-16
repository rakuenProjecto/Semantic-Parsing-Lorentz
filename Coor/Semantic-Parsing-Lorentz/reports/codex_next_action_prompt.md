# Codex Next Action Prompt

- cycle: `191`
- last status: `dry_run_planned`
- last commit: `ae553add13be20ab8bc94e014e690acdd3c34ced`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
