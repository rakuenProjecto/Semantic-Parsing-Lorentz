# Codex Next Action Prompt

- cycle: `128`
- last status: `dry_run_planned`
- last commit: `f3cb1ea38e56362303db3a2469d6fb9c483e689b`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
