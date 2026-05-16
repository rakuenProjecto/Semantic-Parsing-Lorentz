# Codex Next Action Prompt

- cycle: `249`
- last status: `dry_run_planned`
- last commit: `7ec6a1e53b89c7b5e58040bf0a3b03484920893f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
