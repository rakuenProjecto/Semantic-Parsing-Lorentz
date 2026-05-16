# Codex Next Action Prompt

- cycle: `264`
- last status: `dry_run_planned`
- last commit: `16468316fce66af3635bbabd6cacef6dc0f5043c`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
