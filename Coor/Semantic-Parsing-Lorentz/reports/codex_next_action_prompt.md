# Codex Next Action Prompt

- cycle: `214`
- last status: `dry_run_planned`
- last commit: `de894c6266ba4e76ff6fd904b2959796658c1c5f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
