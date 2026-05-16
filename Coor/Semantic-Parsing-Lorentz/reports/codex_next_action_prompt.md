# Codex Next Action Prompt

- cycle: `48`
- last status: `dry_run_planned`
- last commit: `14b3739f7e6e74597fcb783f6c02ed744cbed24c`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
