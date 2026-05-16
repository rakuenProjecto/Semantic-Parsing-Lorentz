# Codex Next Action Prompt

- cycle: `224`
- last status: `dry_run_planned`
- last commit: `ceefe728beae0b7f74800389f35aa4ce13f4a10a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
