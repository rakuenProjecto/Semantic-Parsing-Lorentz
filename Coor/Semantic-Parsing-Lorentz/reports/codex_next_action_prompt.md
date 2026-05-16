# Codex Next Action Prompt

- cycle: `13`
- last status: `dry_run_planned`
- last commit: `c95c5d1370ad975ed9f2e34df31b55a844180e5f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
