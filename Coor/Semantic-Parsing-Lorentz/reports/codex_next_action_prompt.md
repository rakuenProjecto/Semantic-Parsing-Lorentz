# Codex Next Action Prompt

- cycle: `113`
- last status: `dry_run_planned`
- last commit: `d77f9fb5ff1bb900c2b66d16016f91a6f0cdb5f3`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
