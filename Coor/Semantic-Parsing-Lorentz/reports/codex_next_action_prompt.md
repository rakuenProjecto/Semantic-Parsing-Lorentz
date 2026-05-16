# Codex Next Action Prompt

- cycle: `172`
- last status: `dry_run_planned`
- last commit: `a530023e0564836ccc7c0dac36f323be4d031783`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
