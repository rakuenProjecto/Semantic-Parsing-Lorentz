# Codex Next Action Prompt

- cycle: `116`
- last status: `dry_run_planned`
- last commit: `4fb415a25bdbed088dbacf5358745586ec73f993`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
