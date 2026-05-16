# Codex Next Action Prompt

- cycle: `47`
- last status: `dry_run_planned`
- last commit: `54f2335e3043ce78f2e16aaecb4430c1eeef34b1`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
