# Codex Next Action Prompt

- cycle: `138`
- last status: `dry_run_planned`
- last commit: `d4f29d5cee024baa0a9976b95c25cdb0f4df7fd4`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
