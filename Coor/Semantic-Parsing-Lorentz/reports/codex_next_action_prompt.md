# Codex Next Action Prompt

- cycle: `157`
- last status: `dry_run_planned`
- last commit: `94be2dfcdda91bef49ece04a03ac17b57a372d61`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
