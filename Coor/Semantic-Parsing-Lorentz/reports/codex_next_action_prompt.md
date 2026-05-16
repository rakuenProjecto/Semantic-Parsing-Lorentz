# Codex Next Action Prompt

- cycle: `106`
- last status: `dry_run_planned`
- last commit: `04ed3961944ff240193cb8bbc132c22975f37c3b`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
