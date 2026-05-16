# Codex Next Action Prompt

- cycle: `46`
- last status: `dry_run_planned`
- last commit: `0f0801ef611706ba57c210b11e80c600a4c0e858`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
