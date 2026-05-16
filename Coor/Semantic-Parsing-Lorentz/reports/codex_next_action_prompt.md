# Codex Next Action Prompt

- cycle: `200`
- last status: `dry_run_planned`
- last commit: `18e561d93b9d0aecaef154a670f12ffd44d5ff86`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
