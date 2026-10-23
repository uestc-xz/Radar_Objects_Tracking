# my_track numerical stability and measurement anchoring

Date: 2026-10-23. Baseline: `2e1e685`.

User authorization: improve training autonomously, retain sequence tracking rather
than create a separate single-frame model, and record design before editing code.
Implementation uses the writing-plans/TDD workflow in this session.

## Goal and Frozen Contracts

Remove demonstrated optimizer corruption before evaluating architecture quality.
M1 remains frozen; `IS_COLD=True` consumes the existing cache. Keep all 50 queries,
three fixed identity slots, multiple measurements per slot, detached TrackBank
updates, teacher curriculum, loss formulas/weights, LR, and validation definitions.
Do not change existing experiment directories or overwrite previous checkpoints.
Two epochs are a screening budget, not proof of final identity/generalization quality.

## Evidence Before Implementation

- Working tree is clean at baseline.
- One midpoint frame from each of 25 train and 8 validation sequences was checked.
  M1 arrays and GT coordinates share meter units, without a global 3/3.5 m offset.
  Many matched errors are decimeters; individual errors over 1 m and missed high-
  confidence detections remain. This sample is not a full-set accuracy estimate.
- With one observed slot and two empty slots, real JointTracker + tracker_losses
  produces finite loss but pair-head gradients up to `3.60e35`. Squaring those
  float32 gradients overflows, including Adam's second-moment accumulator.
- The existing `original/my_track/runs/teacher_lr3e4_diagnostics/` epoch-1 last
  checkpoint was inspected read-only: 1 of 65 Adam `exp_avg_sq` tensors contains
  nonfinite values. Start with a fresh optimizer; do not resume this state.
- Source: an empty hard-assigned slot divides its zero numerator by float32
  `tiny` (~1.18e-38), while ST softmax retains a derivative through that numerator.
  Existence/confidence heads consume the result even when final_pos is masked.
- A global denominator floor such as 1e-3 is NOT acceptable: it shrinks coordinates
  of genuinely assigned low-score measurements toward the origin.
- At cold start, predicted_pos is zero. A random correction head combines it with
  the real measurement using a roughly half-open gate. This introduces large
  initial coordinate error even with correct teacher assignments.

## Task 1: Empty-slot Gradients and Optimizer Safety

Files: `joint_tracker.py`, `train.py`, `config.py`, joint/training tests.

- [x] RED: force all candidates into slot 1; backpropagate all four losses and run
  Adam; assert finite bounded gradients and finite optimizer moments.
- [x] All-zero scores leave zero fused_pos/feat with finite backward (the old
  implementation also passes this characterization test).
- [x] Keep the existing tiny-positive-score normalization test.
- [x] Implement `observed = sum(weights)>0`; divide observed slots by their exact
  sum, empty slots by 1, then mask empty-slot fused outputs to zero. This blocks
  the nonexistent fusion derivative instead of inventing an epsilon observation.
- [x] RED: gradient diagnostics must report norm 5e30 for [3e30,4e30], not Inf.
- [x] Accumulate norms in float64; reject nonfinite gradients before optimizer.step.
  Clip valid global norms at configurable 5.0 before Adam; log the preclip norm.
- [x] Run focused tests and document results (44 tests passed at this stage).
- [ ] Create a scoped commit and record its hash (sandbox index write denied).

## Task 2: Measurement-anchored Cold Start

Files: `joint_tracker.py`, `test_joint_tracker.py`.

- [x] RED: with correct teacher assignments and default initialization, a new slot
  at (2,4) must output (2,4), not approximately (1,2) plus a random offset.
- [x] Initialize the final correction layer to zero. For an uninitialized observed
  slot use gate=1; initialized observed slots keep the learned sigmoid gate.
  The residual remains learnable and may leave the candidate convex hull.
- [x] Keep history-only missing-observation behavior; test mixed new/old slots and
  nonzero gradients on both association CE and the correction layer.
- [x] Run focused tests (46 passed). No new checkpoint compatibility
  issue: parameter names and shapes are unchanged; initialization affects fresh runs.

## Task 3: Bounded Real-data Screening

Files: `scripts/screen_tracker.py`, `tests/my_track/test_tracker_screening.py`.

- [x] Add a cold-cache-only runner with explicit epoch/data limits and seed.
  Select windows evenly within every training folder, not just the first single-
  person sequences. Preload only the selected windows; num_workers=0.
- [x] Reuse production run_training and evaluator; require an explicitly named NEW
  output directory, refuse overwriting, and write a manifest with exact sample IDs,
  limits, optimizer settings and Git state. Do not mutate config.py defaults.
- [x] Print screening limits and preserve frame order within validation records.
  Truncated validation must be labeled as such, not reported as full-set evaluation.
- [x] Test deterministic selection, argument rejection and overwrite protection.
- [x] Run full unit tests (50 passed), compile/diff checks and real-data two-epoch
  screens without stopping or killing an existing user GPU process.

## Risks and Interpretation

These fixes remove demonstrated numerical failure and avoid destroying good M1
coordinates at initialization. They do not by themselves solve fixed-slot cold-
start symmetry, teacher/autonomous distribution shift, ID switches, or long-gap
motion drift. Do not call low teacher loss a tracking success. Use autonomous
position, foreground association accuracy, coverage, IDSW and out-of-bounds ratio
together. Check sample-ID timing before changing DT (docs conflict with config).

## Results

### Verification and Small Examples

- An empty slot has all normalized weights exactly zero. Both its fused output
  and fusion derivative are zero; no artificial observation is introduced.
- One assigned candidate at (2,4), even with a tiny positive score s, has weight
  s/s=1 and fused position (2,4). Zero-initialized correction outputs the same
  point for a new slot. A slot with history still uses its sigmoid gate.
- Focused tests passed after each change: 44 for numerical safety, 46 including
  anchoring, and finally all 50 with the screening runner. Commands:
  `E:\Miniconda3\python.exe -m unittest discover -s tests/my_track -v`,
  `python -m py_compile original/my_track/joint_tracker.py original/my_track/train.py scripts/screen_tracker.py`,
  and `git diff --check`.
- Teacher-assigned, untrained midpoint-frame cold-start comparison (same seed 0;
  8 validation records; 13 valid observed GT slots): baseline mean Euclidean
  error 1.2776m, anchored initialization 0.3237m. This is NOT autonomous tracking
  evaluation and does not establish a full-set improvement.

### Two-epoch Screens

Both runs use seed 0, 100 windows (4 spread across time per training record),
batch size 4, and the first 200 consecutive frames of all 8 validation records.
Each epoch has 25 optimizer updates. Teacher ratio remains 1.0. These are truncated
screens, not two full production epochs or an old/new controlled comparison.

| LR | Epoch | Train | Val | Val position | MAE m | FG accuracy | Target coverage | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3e-5 | 1 | 2.78711 | 4.54894 | 2.44289 | 1.94787 | 0.01697 | 0.07143 | 1 |
| 3e-5 | 2 | 2.06459 | 4.24746 | 2.24920 | 0.89805 | 0.24185 | 0.50000 | 9 |
| 3e-4 | 1 | 1.59715 | 2.21352 | 0.76593 | 0.94185 | 0.53693 | 0.57143 | 8 |
| 3e-4 | 2 | 1.07513 | 2.53822 | 0.77090 | 0.95665 | 0.53759 | 0.57143 | 10 |

Active-position out-of-bounds ratio is zero in all four rows. Each new epoch-2
checkpoint has 0/65 nonfinite Adam second-moment tensors, versus 1/65 in the old
epoch-1 checkpoint. No old logs/checkpoints or running user process were modified.

Artifacts (ignored by Git) are in
`test_results_tracker_screen_20261023/stable_anchor/` and
`test_results_tracker_screen_20261023/stable_anchor_lr3e4/`, each with CSV,
checkpoints and a manifest containing exact train/validation sample IDs. To repeat,
choose a fresh output directory and run:

```powershell
python scripts/screen_tracker.py --output test_results_tracker_screen/run1 --device cuda
python scripts/screen_tracker.py --output test_results_tracker_screen/run2 --device cuda --lr 3e-4
```

### Remaining Failure and Rejected Diagnostic

On the 3e-4 epoch-2 checkpoint, autonomous validation predicts candidate classes
`[clutter,slot1,slot2,slot3] = [76484,3516,0,0]` across 1600 frames. Observed-slot
frame counts are `[1600,0,0]`. The model collapses to one occupied slot; low
teacher position loss is therefore not evidence of good multi-target tracking.
The standard LR remains 3e-5: the 3e-4 screen's validation loss rises in epoch 2,
and its target coverage remains insufficient.

A read-only hook diagnostic subtracted `lambda * squared_distance_m` from each
initialized slot's pair logit (all candidates retained; uninitialized slots exempt).
It reused the trained 3e-4 checkpoint without retraining and reset banks per run:

| lambda | Val | Val position | MAE m | Coverage | IDSW |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 10.36308 | 1.06021 | 2.24549 | 0.71429 | 13 |
| 1.0 | 75.96105 | 1.39819 | 0.57506 | 1.00000 | 16 |
| 2.0 | 323.59567 | 3.24420 | 6.14145 | 1.00000 | 24 |

This bias is NOT merged into model code. Better coverage alone is not enough:
fixed-GT-slot loss, warm-up-mapped tracking metrics and IDSW disagree. A future
architecture experiment should explicitly handle vacant-slot births, competition
between established tracks and newborn slots, and sequence-consistent identity
supervision. Do not silently add per-frame GT reassignment to conceal switches.

### Scope of the Conclusion

The numerical corruption and cold-start coordinate shrinkage are reproduced,
fixed and regression-tested. General multi-target tracking remains unsolved.
No claim of full validation improvement or of a preferred final LR is made.

## Follow-up: Sequence-local Slot Semantics and Autonomous Curriculum

### Data Finding and Design

Within each sequence, GT slot IDs are stable; across sequences, they are not
global person identities. A training sample resets TrackBank at each 50-frame
window, so train slots are now canonicalized once per window by each target's
first valid `(x,y)` coordinate (x primary, y tie-break), with absent slots last.
Validation applies the same ordering once to each full recording. The resulting
mapping never changes inside the window/record, including when targets cross.
This gives slots a sequence-local, observable initialization convention without
per-frame Hungarian relabeling. GT-derived association labels, teacher fusion,
position and existence targets all use the same reordered tensor. Metrics still
use one fixed mapping over each validation record and retain their formulas.

The screened training candidate-label counts were `[clutter,slot1,slot2,slot3] =
[238729,5525,3802,1944]`; validation prefix counts were `[76994,1719,817,470]`.
An optional per-slot-balanced foreground CE was tested, but removed: compared
with canonicalization alone it did not improve foreground accuracy or coverage
in the two-epoch screen, and would add a second association-loss formula.

### Curriculum Screen

`screen_tracker.py` now accepts optional `--teacher-warmup-epochs` and
`--teacher-anneal-epochs` overrides, recorded in its manifest; normal config
defaults remain unchanged. A regression test catches Python's captured default
arguments so explicit schedule overrides actually reach training.

Same 100 windows, seed 0, batch 4, 3e-5 LR, 8 validation prefixes of 200 frames;
the schedule was warmup 0 / anneal 2, hence teacher ratios `[0.5,0.0]`:

| Epoch | Teacher | Train | Val | Val position | MAE m | FG accuracy | Coverage | OOB | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5 | 3.64028 | 13.34198 | 11.00769 | 28.793 | 0.279 | 1.000 | 0.518 | 12 |
| 2 | 0.0 | 15.90956 | 6.89888 | 4.61851 | 10.314 | 0.282 | 1.000 | 0.509 | 10 |

This is not a usable model: all-slot coverage coexists with large position error
and over half of active positions outside the physical region. In epoch 2 the
mean pre-clip gradient norm is `976.220` and train position loss `2.73139`.
The loss then falls within validation but remains far above the teacher screens;
this is not evidence of convergence. The experiment was allowed to finish both
epochs. Its outputs and manifest are under
`test_results_tracker_screen_20261023/autonomous_fixed_lr3e5/`.

Canonicalization-only two-epoch results (before dropping the ineffective balance
option) were: LR=3e-5, epoch 2 val `4.23289`, MAE `0.89887m`, foreground accuracy
`0.21158`, coverage `0.50`; LR=3e-4, epoch 2 val `2.46853`, MAE `0.95683m`,
foreground accuracy `0.57186`, coverage `0.57143`. Relative to the corresponding
non-canonical screens, only small metric changes occurred; all foreground
candidates still chose slot 1 (`[76486,3514,0,0]`). Therefore neither sequence
canonicalization nor per-slot balancing alone solves autonomous multi-target
tracking. Canonicalization remains as a label-semantics correction; balancing is
not retained.

### Next Engineering Conclusion

The current all-or-nothing training fusion modes expose a stability gap: teacher
ratio 1.0 yields good supervised state updates but lets prediction errors remain
outside the feedback loop; dropping to 0.5 makes false births feed back and
corrupt later state. Future work should focus on cautious track birth and
observation gating (especially for uninitialized slots), rather than another
global distance bias or abrupt teacher-ratio change. Keep the current production
teacher schedule until a separately tested birth/confirmation mechanism can
reject false candidates without suppressing genuine secondary tracks.

## Follow-up: M1 Decoder Embedding Ablation

The 128-D tensor passed from M1 is the final decoder query embedding. M1 trains it
through coordinate and foreground/background heads, but has no explicit
cross-frame identity-consistency objective. To test its utility without changing
the model parameterization, `JointTracker(use_m1_features=False)` zeros both
current measurement embeddings and stored track embeddings before use. This
option is not part of the state dict; the default remains enabled, so existing
checkpoints load unchanged. `screen_tracker.py --disable-m1-features` provides a
matched train-time ablation, and `scripts/compare_m1_features.py` compares modes
on a fixed checkpoint as an inference-dependence diagnostic only.

Matched screen: seed 0, same 100 training windows, batch 4, LR `3e-4`, two
epochs, and eight validation records capped at 200 frames. The selected windows
and validation records are identical in both manifests:

| M1 embedding | Epoch-2 train | Val loss | Val position | MAE m | FG accuracy | Coverage | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| On | 1.07522 | 2.46854 | 0.81225 | 0.95683 | 0.57186 | 0.57143 | 10 |
| Off | 0.99992 | 2.48522 | 0.94775 | 0.97718 | 0.54391 | 0.57143 | 10 |

The feature-on screen is slightly better on position, MAE and foreground
association accuracy, but the margins are small and coverage/IDSW are unchanged.
This short screen does not establish that embeddings are useful or harmful; it
does show that removing them is not a clear fix for the dominant slot-collapse
problem. The same-checkpoint inference ablation is strongly out-of-distribution:
the old checkpoint routes all detections to clutter when embeddings are zeroed,
so it is only evidence that the trained model depends on this input, not an
apples-to-apples measure of its value. Screen outputs are under
`test_results_tracker_screen_20261023/features_on_matched/` and
`test_results_tracker_screen_20261023/features_off_matched/`.

## Follow-up: Frame-local M1 Clustering Baseline

To separate candidate localization from identity management, an inference-only
baseline thresholds cached M1 scores, joins candidates whose Euclidean distance
is within a radius (connected components), and computes one score-weighted
centroid per component. It does not read GT while constructing points. Evaluation
matches each frame's predicted point set to nonempty GT with Hungarian matching;
pairs beyond 1m count as false positive plus false negative. There is no temporal
state or ID-switch metric.

The complete validation set (8 records, 17,421 frames) produced:

| Score threshold | Cluster radius | Precision | Recall | F1 | Matched MAE m | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.96 | none (raw candidates) | 0.9139 | 0.8629 | 0.8877 | 0.2767 | 2628 | 4432 |
| 0.96 | 0.25 m | 0.9176 | 0.8629 | 0.8894 | 0.2769 | 2505 | 4432 |
| 0.96 | 0.50 m | 0.9231 | 0.8628 | 0.8919 | 0.2772 | 2325 | 4435 |
| 0.96 | 0.75 m | 0.9283 | 0.8624 | 0.8942 | 0.2778 | 2153 | 4448 |

The baseline indicates M1 has useful per-frame localization: at score 0.96 the
matched point error is about 0.28m, and clustering mostly reduces duplicate
candidate false positives while leaving recall and position error nearly
unchanged. This supports focusing the next learned experiment on candidate-to-
target assignment and identity continuity, not starting with a major coordinate
regression redesign. It does not imply that 13.8% of GT frames are model false
negatives globally in a temporal sense; this is strictly framewise point-set
matching under a 1m tolerance.

The evaluation matcher uses dummy assignments so it maximizes the number of
in-radius matches before minimizing their total distance. A regression test
covers a geometric case where unconstrained minimum-cost matching would reject
one otherwise valid pair. Threshold/radius sweeps on validation are exploratory;
the best grid entry is not a held-out generalization estimate.

Reproduction:

```powershell
python scripts/evaluate_m1_cluster_baseline.py --max-frames 0 --output test_results_tracker_screen_20261023/m1_cluster_baseline_full.json
```

The JSON output is ignored by Git with other screening artifacts.
