# Codex Next Action Prompt

- cycle: `254`
- last status: `dry_run_planned`
- last commit: `7e67bedaf9ae949649d6cd18d40fb472b53cc4e9`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
