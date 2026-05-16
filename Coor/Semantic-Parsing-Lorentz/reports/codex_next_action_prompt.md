# Codex Next Action Prompt

- cycle: `107`
- last status: `dry_run_planned`
- last commit: `228736f0a694a22fabe28145b0c371660282fbb5`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
