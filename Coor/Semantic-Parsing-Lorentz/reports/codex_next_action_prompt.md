# Codex Next Action Prompt

- cycle: `147`
- last status: `dry_run_planned`
- last commit: `16e873d9c138cb4641ee858eaa40b0cfce0cabfc`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
