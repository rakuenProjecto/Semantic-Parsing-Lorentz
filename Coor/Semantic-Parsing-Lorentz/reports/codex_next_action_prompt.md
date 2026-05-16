# Codex Next Action Prompt

- cycle: `187`
- last status: `dry_run_planned`
- last commit: `ac424990943fe6516a70859e290c9318ef36c2e9`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
