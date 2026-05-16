# Codex Next Action Prompt

- cycle: `248`
- last status: `dry_run_planned`
- last commit: `bf1d3afbff7f8e1ee68f6eb4c1895d9f0ed94448`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
