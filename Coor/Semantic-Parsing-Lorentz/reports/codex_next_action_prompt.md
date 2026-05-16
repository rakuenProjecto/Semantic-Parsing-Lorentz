# Codex Next Action Prompt

- cycle: `164`
- last status: `dry_run_planned`
- last commit: `a429e3b1be8a0cb710144f35c88ae1650a1b4653`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
