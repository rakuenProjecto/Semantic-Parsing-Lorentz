# Codex Next Action Prompt

- cycle: `105`
- last status: `dry_run_planned`
- last commit: `28d521635fc820dfe86fdd6a2215300cda8cf4e9`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
