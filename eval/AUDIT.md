# Eval audit (eval-audit skill, 2026-10-03)

Routed by evals-start: an eval pipeline exists (scramble-hard, Microban OOD, CI regression gate), so `eval-audit`. Sokoflow has no LLM. The model is a small diffusion policy and every check is a game-engine fact, so the judge and human-review areas mostly do not apply. Findings by impact.

## 1. Error analysis

### Solve rates were reported without looking at why puzzles fail
**Status:** Fixed in this branch.
The committed results only say solved or not. `eval/failure_breakdown.py` replays the 225 diffusion failures through the demo solver's autopsy:
- 219 stop with no new legal state.
- The model's first illegal proposal is walking into a wall (182), pushing a box into a wall (33), or pushing a box into a box (9).
- Most boards end with 2 or 3 boxes still off target.
This points at the model not representing walls and blocked pushes, which is a training-data and conditioning question, not a sampler budget question.

## 5. Labeled data / measurement noise

### Headline from one seed
**Status:** Problem exists (documented).
The demo solver solved 6 of the 225 fast-path failures, so the solve rate moves by a few puzzles between sampling runs. The CI gate stays exact because the run is seeded. Product claims should quote 6.2% as one-seed.
**Fix:** run scramble-hard over 3 to 5 seeds and report the range (about 70 s per seed on an idle CPU).

### Real-level coverage is small
**Status:** Problem exists. Microban OOD is 38 real levels (diffusion 1/38). It is enough to show the gap, but too few to measure progress.

## 2. Evaluator design

**Status:** OK. Solved means the engine reached the goal state: binary and code-checked. BFS is the baseline. No similarity metrics.

## 3. Judge validation / 4. Human review

**Status:** Not applicable (no judge, no subjective labels).

## 6. Pipeline hygiene

**Status:** OK. CI reruns all 240 puzzles and fails on any change in outcome or path length. Intended changes need a new committed result file and a reason in the commit message.
