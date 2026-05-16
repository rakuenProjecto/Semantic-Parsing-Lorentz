# Codex Next Action Prompt

- cycle: `121`
- last status: `dry_run_planned`
- last commit: `a6296ee0ce2b8bd8dbc70830a94a2706d8b416ce`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
