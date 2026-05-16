# Codex Next Action Prompt

- cycle: `114`
- last status: `dry_run_planned`
- last commit: `518d2ff05e15dc22ae7c13f76f5584b7cc6a23f3`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
