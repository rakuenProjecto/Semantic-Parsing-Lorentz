# Codex Next Action Prompt

- cycle: `41`
- last status: `dry_run_planned`
- last commit: `5bc5963e541f7e2deef3e304fabbb09e08382836`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
