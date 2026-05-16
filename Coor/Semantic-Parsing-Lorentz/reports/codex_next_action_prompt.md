# Codex Next Action Prompt

- cycle: `100`
- last status: `dry_run_planned`
- last commit: `6c8a410af6e1c65ba6f5a176fae12494f7099a86`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
