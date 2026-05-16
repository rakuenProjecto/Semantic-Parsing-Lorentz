# Codex Next Action Prompt

- cycle: `222`
- last status: `dry_run_planned`
- last commit: `f1ca68fa9cc2390e89348b2c86252dae0e9b2052`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
