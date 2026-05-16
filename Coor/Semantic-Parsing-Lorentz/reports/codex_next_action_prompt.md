# Codex Next Action Prompt

- cycle: `145`
- last status: `dry_run_planned`
- last commit: `573835834bc09252a91c5d7cd5043ec0c46e2f33`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
