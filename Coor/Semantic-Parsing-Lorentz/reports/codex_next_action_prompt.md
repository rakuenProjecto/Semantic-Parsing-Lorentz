# Codex Next Action Prompt

- cycle: `220`
- last status: `dry_run_planned`
- last commit: `ee6b78f5dcf049f6eeccb70779ad56c52778cfce`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
