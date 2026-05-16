# Codex Next Action Prompt

- cycle: `232`
- last status: `dry_run_planned`
- last commit: `4a0cc70a841a1f80d7dd1d8a755e80507dff9b7b`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
