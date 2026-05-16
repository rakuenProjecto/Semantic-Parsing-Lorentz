# Codex Next Action Prompt

- cycle: `168`
- last status: `dry_run_planned`
- last commit: `adb63c766e2951a4add470086453f2ac29a45b85`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
