# Codex Next Action Prompt

- cycle: `72`
- last status: `dry_run_planned`
- last commit: `f4461ab0448668d3e4a6fceb2a398db8433837d4`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
