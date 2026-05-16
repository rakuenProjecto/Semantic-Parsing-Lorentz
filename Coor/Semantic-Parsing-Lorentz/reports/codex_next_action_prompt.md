# Codex Next Action Prompt

- cycle: `126`
- last status: `dry_run_planned`
- last commit: `54b5fe2dffea59d548e1bfc937fd0d2e0129ecdb`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
