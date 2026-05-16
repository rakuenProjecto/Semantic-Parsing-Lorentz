# Codex Next Action Prompt

- cycle: `215`
- last status: `dry_run_planned`
- last commit: `9444822de86f27a6f3f90b9727d6679d894072af`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
