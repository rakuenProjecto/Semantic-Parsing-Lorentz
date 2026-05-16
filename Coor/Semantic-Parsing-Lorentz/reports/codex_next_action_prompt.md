# Codex Next Action Prompt

- cycle: `154`
- last status: `dry_run_planned`
- last commit: `a61a2b1367c355fffdbdd1752702d410c54bc04a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
