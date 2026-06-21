# Current Goal Guardrail: Downstream V3 First-Version Fine-Tune

Last updated: 2026-06-21

This file is the durable execution contract for the active goal. If the thread
is resumed after a long run or context compaction, start from this file before
deciding whether the goal is complete.

## Scope lock

This goal was intentionally tightened after the fast downstream queue had
already started. Do not relax it later based on memory, partial context, or the
fact that the queue has been running for a long time.

When resuming this thread, use the following order of truth:

1. The active Codex goal text.
2. This guardrail file.
3. `docs/downstream_v3_plan.md`.
4. The actual queue/config/output state on disk.

The active goal must not be marked complete just because training processes
started, because the PD random split finished, or because the queue supervisor
reports no running jobs. Completion requires the hard audit, result table, test
run, artifact check, and documentation updates listed below.

Before any `update_goal(status="complete")` call, run and pass:

1. `python scripts/audit_downstream_v3_goal.py --status-json outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast_clean_resume/status.json --materialize-plan-artifacts --report-json outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast_clean_resume/goal_audit_final_materialized.json`
2. `python scripts/audit_downstream_v3_goal.py --status-json outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast_clean_resume/status.json --report-json outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast_clean_resume/goal_audit_final.json`
3. `python scripts/summarize_downstream_v3_results.py --require-complete`
4. `python scripts/summarize_downstream_v3_convergence.py --view within30 --require-complete`
5. `python scripts/summarize_downstream_v3_convergence.py --view epoch1 --require-complete`
6. `pytest -q tests`

If any command fails, any required artifact is missing, or any of the 32 jobs is
not represented in the final summary, keep the goal active and fix the issue.

## Non-negotiable completion gate

The current goal is not complete until all requirements below are satisfied.

1. The run must satisfy `docs/downstream_v3_plan.md` first-version downstream
   scope.
2. All 8 downstream tasks must finish:
   - `pd_related_5class` using `PD相关_random_seed20260620`
   - `pd_binary` using `PD相关_binary_random_seed20260620`; label 1 merges
     all four PD-related subtypes
   - `epilepsy_binary`
   - `detox_binary`
   - `migraine_binary`
   - `ad_binary`
   - `mci_original_only_binary`: original `MCI` rows only, with
     `source_dataset=匹配后` removed
   - `mci_matched_binary_random_seed20260621`: `MCI匹配后` rows kept only when
     raw `subject` exists in the original `MCI` label anchor; matched labels are
     ignored and overwritten from original `MCI`
     - The label anchor is built only from original `MCI` rows after removing
       `source_dataset=匹配后`. `MCI匹配后` is never a label source: not its
       `health_label`, not its label-like `source_group`. Rows whose raw
       `subject` is absent from that original-MCI anchor must be dropped.
3. Each task must finish all 4 first-version modes:
   - `scratch`
   - `linear_probe`
   - `partial`
   - `full`
4. This means 32 final jobs total. Starting the queue is not enough.
5. Final outputs must come from `outputs/downstream_v3_fast/`, not from the
   older baseline root `outputs/downstream_v3/`.
6. PD 5-class must use the new subject-level stratified random split:
   `data-260606/extracted/eyemae_fast_dataset_v1/downstream/PD相关_random_seed20260620/{train,validation,test}.csv`.
7. PD binary must reuse the same subject split through:
   `data-260606/extracted/eyemae_fast_dataset_v1/downstream/PD相关_binary_random_seed20260620/{train,validation,test}.csv`.
8. Epilepsy, detox, migraine, and AD use the formal train/validation/test split
   directories already present in `eyemae_fast_dataset_v1`; the two MCI tasks
   use the corrected generated views above.
9. The fast fine-tune policy must be used for this queue:
   `max_epochs=100`, `early_stopping_patience=10`,
   `min_epochs_before_early_stopping=0` for all 4 modes, including scratch.
10. Best checkpoint selection must be based on validation subject-level main
   metric, and each final report must include train, validation, and test
   metrics after loading the best checkpoint.
11. Final reporting must also include convergence-speed views:
    - test results for the best validation checkpoint within epochs 0-29
      when available or rerunnable
    - test results after exactly one training epoch for each task/mode
12. The final audit must cover the first-version tests and acceptance metrics
    in `docs/downstream_v3_plan.md`, especially Sections 21-24.
13. No subject overlap is allowed between train/validation/test for any task.
14. Documentation must be updated through completion:
    - `docs/newdata_v3_training_log.md`
    - `docs/newdata_v3_implementation_changes.md`
    - `docs/newdata_v3_exception_log.md`

## Active queue

- Queue list: `configs/downstream/queue_pd_random_seed20260620_fast.txt`
- Queue log dir: `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast/`
- Queue output root: `outputs/downstream_v3_fast/`
- tmux session: `eyemae_downstream_v3_fast`
- Tail tmux session: `eyemae_downstream_v3_tail32`; waits for the current
  queue to become idle, then resumes the expanded 32-job config list with
  `--resume-status` so the newly added PD binary jobs run without duplicating
  completed jobs.
- Clean-resume tail tmux session: `eyemae_downstream_v3_tail32_allowfail`.
  This is the authoritative tail queue for final completion. It waits for the
  current queue to become idle, tolerates the known upstream `rc=-15` from the
  retracted scratch cap experiment, then launches the expanded 32-job config
  list under
  `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast_clean_resume/`.
- within30 lock tmux session: `eyemae_downstream_v3_within30_lockwatch`.
  It runs `scripts/lock_downstream_v3_within30_checkpoints.py` once per minute
  while the queue is active, preserving epochs 0-29 best checkpoints for jobs
  launched before automatic within30 checkpointing was added.
- GPUs: `1,2,3,4`

## Completion audit checklist

Before marking the goal complete, verify:

1. `outputs/downstream_v3_fast_logs/run_20260620_random_pd_fast_clean_resume/status.json`
   reports all 32 queue jobs completed and no running/pending/failed jobs. The
   original queue status under `run_20260620_random_pd_fast/` may retain the
   documented `epilepsy_binary_scratch` `rc=-15` interruption and is not the
   final completion status.
2. There are exactly 32 `metrics_final.json` files under
   `outputs/downstream_v3_fast/` for the required task x mode matrix.
3. Each `metrics_final.json` contains final train, validation, and test metrics.
4. Each completed experiment has the first-version output artifacts required
   by `docs/downstream_v3_plan.md` Section 23, or a documented equivalent plus
   a materialized compatibility artifact:
   - `resolved_config.yaml`
   - `run_summary.json`
   - `checkpoint_last.pt`
   - `checkpoint_best.pt`
   - final split metrics for train, validation, and test
   - trial and subject prediction CSVs for validation and test
   - confusion matrices for validation and test
5. The result table records the first-version acceptance metrics from
   `docs/downstream_v3_plan.md` Section 24:
   - pretrain exposure
   - encoder initialization
   - fine-tune mode
   - subject AUROC or macro AUROC
   - subject AUPRC or macro AUPRC
   - balanced accuracy
   - F1
   This table must be generated by `scripts/summarize_downstream_v3_results.py
   --require-complete` after all jobs finish.
   The two convergence-speed tables must also be generated by
   `scripts/summarize_downstream_v3_convergence.py --view within30
   --require-complete` and
   `scripts/summarize_downstream_v3_convergence.py --view epoch1
   --require-complete`, after the required early checkpoints/evaluations or
   documented reruns are available.
6. The available tests covering `docs/downstream_v3_plan.md` Section 21 have
   been run and recorded. If a named plan test does not exist under `tests/`,
   record the closest current test coverage and the gap explicitly.
7. The PD 5-class configs in the completed queue point to
   `PD相关_random_seed20260620`.
8. The PD binary configs in the completed queue point to
   `PD相关_binary_random_seed20260620`.
9. The six non-PD tasks point to their formal split directories.
10. The final training log has a compact result table for all 32 jobs.
11. The exception log records any interruption/restart/failure recovery that
   happened during the run.

If any item above is missing, continue monitoring or fix the run; do not mark
the active goal complete.
