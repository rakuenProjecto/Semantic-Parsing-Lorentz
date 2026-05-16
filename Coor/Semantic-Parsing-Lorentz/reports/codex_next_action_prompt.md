# Codex Next Action Prompt

- cycle: `221`
- last status: `dry_run_planned`
- last commit: `f38a2908280294bd09fc307c408731de6afa2191`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
