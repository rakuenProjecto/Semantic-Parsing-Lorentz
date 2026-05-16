# Codex Next Action Prompt

- cycle: `51`
- last status: `dry_run_planned`
- last commit: `0470f6b87bbf8932bdc90561ad83fb8b748608cb`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
