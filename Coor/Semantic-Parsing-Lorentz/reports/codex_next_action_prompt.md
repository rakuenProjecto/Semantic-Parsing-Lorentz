# Codex Next Action Prompt

- cycle: `49`
- last status: `dry_run_planned`
- last commit: `986e1ba3a366328f10e93a860d74404e305b4abe`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
