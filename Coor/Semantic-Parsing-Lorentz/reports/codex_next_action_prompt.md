# Codex Next Action Prompt

- cycle: `175`
- last status: `dry_run_planned`
- last commit: `98bb2d2ff1e3ff5fe47e87680e4fc3f4f8c733bf`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
