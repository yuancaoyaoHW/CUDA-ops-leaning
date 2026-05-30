# Migration scripts

One-shot scripts that mutate `progress.yaml` schema. Run once, then ignore.

## migrate_yaml.py

Phase A schema upgrade (2026-05-30 spec):

- adds `meta.verify_defaults`
- adds `operators.<op>.kind` (manual mapping)
- adds `week0.day00` warmup gate
- adds `weekN.dayNN.operator/phase`
- adds `star_filled / algo_drill_done / cpp_drill_done` on weekly check days

Idempotent. Run with `--dry-run` first.
