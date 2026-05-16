# Codex Next Action Prompt

- cycle: `118`
- last status: `dry_run_planned`
- last commit: `11b4dfe13914e0fa163c3689fc35cb736c74e3f4`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
