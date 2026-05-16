# Codex Next Action Prompt

- cycle: `159`
- last status: `dry_run_planned`
- last commit: `69adc948af4d1ab5db5e94f0779e44518d2b0ea7`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
