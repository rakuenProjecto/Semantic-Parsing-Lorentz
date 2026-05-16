# Codex Next Action Prompt

- cycle: `181`
- last status: `dry_run_planned`
- last commit: `6914e44e0780e6bd5401205a914af4076b99f992`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
