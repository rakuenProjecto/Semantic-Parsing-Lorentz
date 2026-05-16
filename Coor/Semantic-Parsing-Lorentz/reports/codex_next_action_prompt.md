# Codex Next Action Prompt

- cycle: `63`
- last status: `dry_run_planned`
- last commit: `4c44f0e5111971f2217040bc5a866685f77e40b4`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
