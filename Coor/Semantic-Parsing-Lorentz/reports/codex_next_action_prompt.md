# Codex Next Action Prompt

- cycle: `42`
- last status: `dry_run_planned`
- last commit: `90dfa2ad23dbe408fb35e631cbf6e62660d45856`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
