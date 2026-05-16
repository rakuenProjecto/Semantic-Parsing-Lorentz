# Codex Next Action Prompt

- cycle: `163`
- last status: `dry_run_planned`
- last commit: `1b9733da3dac78ef86076edc45577057869ce7e4`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
