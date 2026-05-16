# Codex Next Action Prompt

- cycle: `250`
- last status: `dry_run_planned`
- last commit: `46b67507ac8db4be1c8ba3a24ebe5d6da2a534dc`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
