# Codex Next Action Prompt

- cycle: `73`
- last status: `dry_run_planned`
- last commit: `f77d82cdcd674a0442d27a31a0d5e769fb51db58`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
