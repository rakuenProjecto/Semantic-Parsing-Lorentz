# Codex Next Action Prompt

- cycle: `201`
- last status: `dry_run_planned`
- last commit: `1e4eaf105bb20b8e18d22dce56dc81768af61431`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
