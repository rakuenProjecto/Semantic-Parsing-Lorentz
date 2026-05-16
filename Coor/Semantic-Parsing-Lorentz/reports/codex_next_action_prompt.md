# Codex Next Action Prompt

- cycle: `197`
- last status: `dry_run_planned`
- last commit: `a24c01f3b48144c07d3d16c83264bb27d3ead064`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
