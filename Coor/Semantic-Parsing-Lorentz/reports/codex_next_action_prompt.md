# Codex Next Action Prompt

- cycle: `44`
- last status: `dry_run_planned`
- last commit: `b495cc56baf21a3389e2e36e6094ece21bcf3959`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
