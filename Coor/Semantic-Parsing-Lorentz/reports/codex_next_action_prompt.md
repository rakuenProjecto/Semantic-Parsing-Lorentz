# Codex Next Action Prompt

- cycle: `149`
- last status: `dry_run_planned`
- last commit: `38befa24a537a253fb61beb81fb3b3abeba4d51a`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
