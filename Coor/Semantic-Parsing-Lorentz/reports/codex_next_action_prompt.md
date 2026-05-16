# Codex Next Action Prompt

- cycle: `43`
- last status: `dry_run_planned`
- last commit: `d6550b35b016ca0f376b571a8f8e45fe1c75db26`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
