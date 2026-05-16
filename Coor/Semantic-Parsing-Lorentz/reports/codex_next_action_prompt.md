# Codex Next Action Prompt

- cycle: `10`
- last status: `dry_run_planned`
- last commit: `cd85c1e3820788b773c43a99f92148faeaf09ad7`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
