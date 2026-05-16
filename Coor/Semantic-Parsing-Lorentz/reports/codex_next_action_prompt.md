# Codex Next Action Prompt

- cycle: `198`
- last status: `dry_run_planned`
- last commit: `37a060c18bb9fd5553ff1ed6625dc3eede7e1e83`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
