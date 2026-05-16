# Codex Next Action Prompt

- cycle: `261`
- last status: `dry_run_planned`
- last commit: `92d903ba8a32131c2da65844deff418d7d9e6328`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
