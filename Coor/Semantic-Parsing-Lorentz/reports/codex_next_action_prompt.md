# Codex Next Action Prompt

- cycle: `193`
- last status: `dry_run_planned`
- last commit: `621fad33e81804cf15920ea0963c426c8f49bc6d`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
