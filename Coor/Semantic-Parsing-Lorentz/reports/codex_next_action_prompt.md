# Codex Next Action Prompt

- cycle: `83`
- last status: `dry_run_planned`
- last commit: `333e7dd55daf4b347051b963f3b1b6e858b3ccd3`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
