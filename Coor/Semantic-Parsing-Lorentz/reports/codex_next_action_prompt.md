# Codex Next Action Prompt

- cycle: `216`
- last status: `dry_run_planned`
- last commit: `2d017ae17528252c489df900f1cde654e15ef657`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
