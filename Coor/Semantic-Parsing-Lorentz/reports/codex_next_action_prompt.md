# Codex Next Action Prompt

- cycle: `67`
- last status: `dry_run_planned`
- last commit: `57e6cd687dce1fc618e62b6759c711081d056425`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
