# Codex Next Action Prompt

- cycle: `186`
- last status: `dry_run_planned`
- last commit: `8160778e901d5517cbc260c673b1637135b2c2b9`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
