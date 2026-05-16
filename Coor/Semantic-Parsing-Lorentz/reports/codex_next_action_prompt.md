# Codex Next Action Prompt

- cycle: `192`
- last status: `dry_run_planned`
- last commit: `2b5068fd88d70926ac8ba7a82ae849d8885b850f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
