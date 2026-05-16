# Codex Next Action Prompt

- cycle: `5`
- last status: `dry_run_planned`
- last commit: `6c24a7f0e42b9e6e16122823a4085ce732560439`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
