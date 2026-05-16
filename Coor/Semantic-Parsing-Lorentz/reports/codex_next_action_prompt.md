# Codex Next Action Prompt

- cycle: `180`
- last status: `dry_run_planned`
- last commit: `f1190ef839984d1627084f53c7bf236151326c28`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
