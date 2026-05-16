# Codex Next Action Prompt

- cycle: `85`
- last status: `dry_run_planned`
- last commit: `1d18d03d618934aa44af8b31ee48d1109538a836`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
