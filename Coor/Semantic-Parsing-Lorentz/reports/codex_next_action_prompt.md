# Codex Next Action Prompt

- cycle: `225`
- last status: `dry_run_planned`
- last commit: `ab1e5ce4824a4ae7e6fdd4661ce8104360b396ac`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
