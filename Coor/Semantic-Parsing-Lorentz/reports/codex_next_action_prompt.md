# Codex Next Action Prompt

- cycle: `57`
- last status: `dry_run_planned`
- last commit: `a194e7a52d4b5f2237b2a5ff420bb3973df95de6`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
