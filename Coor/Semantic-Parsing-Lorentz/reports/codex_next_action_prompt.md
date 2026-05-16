# Codex Next Action Prompt

- cycle: `271`
- last status: `dry_run_planned`
- last commit: `91bfb80095a18ad43995cd98c7ef00846e85055d`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
