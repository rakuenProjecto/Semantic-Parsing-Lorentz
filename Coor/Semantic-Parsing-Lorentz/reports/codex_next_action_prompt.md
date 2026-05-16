# Codex Next Action Prompt

- cycle: `75`
- last status: `dry_run_planned`
- last commit: `1ebfa8bf80c5c048d1e064d1979ea6ca90ef5d4f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
