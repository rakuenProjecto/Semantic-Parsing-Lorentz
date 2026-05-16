# Codex Next Action Prompt

- cycle: `265`
- last status: `dry_run_planned`
- last commit: `b58fbdaf708e9e5bc43147199304c27f65c01786`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
