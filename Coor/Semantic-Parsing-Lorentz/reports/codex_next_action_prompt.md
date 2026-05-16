# Codex Next Action Prompt

- cycle: `140`
- last status: `dry_run_planned`
- last commit: `84c8bf6d384c1a61aafd7e5e80558308e62d350d`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
