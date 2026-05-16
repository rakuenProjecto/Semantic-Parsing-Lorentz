# Codex Next Action Prompt

- cycle: `82`
- last status: `dry_run_planned`
- last commit: `8770d7e83b22ce2da220e25a740a961cd95896cd`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
