# Codex Next Action Prompt

- cycle: `256`
- last status: `dry_run_planned`
- last commit: `5bc71c685abf30f9a4a07bda21a2ecfee8f7bdf3`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
