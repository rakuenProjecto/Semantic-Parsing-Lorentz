# Codex Next Action Prompt

- cycle: `59`
- last status: `dry_run_planned`
- last commit: `ba02c696e18ce7f4898c4be7eba461a2dfa3a3e5`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
