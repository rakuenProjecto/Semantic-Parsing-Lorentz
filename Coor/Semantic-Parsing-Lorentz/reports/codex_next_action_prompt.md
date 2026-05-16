# Codex Next Action Prompt

- cycle: `247`
- last status: `dry_run_planned`
- last commit: `9d6fae3193a2eaa967934ae5e7f7489c67e99fcc`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
