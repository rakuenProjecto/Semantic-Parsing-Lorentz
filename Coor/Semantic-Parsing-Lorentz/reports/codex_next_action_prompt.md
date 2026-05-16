# Codex Next Action Prompt

- cycle: `146`
- last status: `dry_run_planned`
- last commit: `a2eac065f68bed74ef231f6e3b095a05ea204aa5`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
