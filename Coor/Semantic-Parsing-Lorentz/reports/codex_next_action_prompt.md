# Codex Next Action Prompt

- cycle: `227`
- last status: `dry_run_planned`
- last commit: `97fb7052fdcb0f94090c2655ae63adf18d7d2971`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
