# Codex Next Action Prompt

- cycle: `92`
- last status: `dry_run_planned`
- last commit: `21217c4c77ba7bc72e1237d1fe8311d66725c68e`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
