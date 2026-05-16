# Codex Next Action Prompt

- cycle: `155`
- last status: `dry_run_planned`
- last commit: `c39bd8862b24deeb3ae0583eee1dd27e5eb1e841`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
