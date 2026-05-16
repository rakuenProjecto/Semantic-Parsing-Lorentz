# Codex Next Action Prompt

- cycle: `207`
- last status: `dry_run_planned`
- last commit: `c3f5386331793bbc1be43892e8d79a109b95389f`

## Diagnosis
[]

## Changes This Cycle
- dry-run only

## Constraints
- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.
- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.
