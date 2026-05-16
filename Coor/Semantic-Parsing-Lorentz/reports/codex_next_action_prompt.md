# Codex Next Action Prompt

- cycle: `15`
- last status: `dry_run_planned`
- last commit: `48786ecb8176e4743dc0de46a279e6f2ba5699cc`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
