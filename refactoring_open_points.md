# Open refactoring points

A review of the fork's own code for small changes that make it more readable
and logical. Written on 2026-11-14, as of commit `abcbba3`. This file is a
working note, not part of the thesis.

## How to continue

- Go through the points one at a time, in the order below unless told otherwise.
- For each point: first explain it again (what is there, why it is a problem,
  the proposed fix). Edit only after "make that change".
- After an edit, run the tests (`./venv/bin/python -m unittest discover -s tests`)
  and check the changed behaviour directly. Leave the edit uncommitted for review.
- When a point changes something the thesis, `README.md` or `CLAUDE.md`
  describes, update that text in the same step. Each point below says whether
  it does.
- Line numbers are as of commit `abcbba3` and will shift. Search for the named
  function or string instead.

Scope: `run_api.py`, `src/api`, `src/services`, `src/queue`, the fork's parts
of `src/bandit`, `config/`, `scripts/` and how `tests/` use them. GreenServ's
inherited parts (`experiments/`, `data/`, `db/`, `src/feature_extractor/`,
`src/metrics/`) are out of scope.

## Done

| Point | What | Commit |
|---|---|---|
| A1 | Removed the in-memory fallback from `RedisClient` | `b0a8108` |
| A2 | Removed `wait_for_result`, the `timeout` branch of `get_result`, `default_queue` | `3c652b1` |
| A3 | Removed `Orchestrator.get_metrics()` and `model_usage_counts` | `0a3ac00` |
| A4 | Removed three unused arguments of `process_query` | `fa93878` |
| A5 | Removed task and result fields nobody reads | `abcbba3` |
| A6 | Removed the used/total fallback for free GPU memory, and the two telemetry fields it read | `778b290` |
| A7 | Removed the telemetry `try/except` and `or []` (orchestrator, controller), the `TypeError` retry and the `features` guards in `RouterService` | `6de24ca` |
| A8 | Removed `task_type` from `EvaluationService._evaluate` and `type: "general"` from `config/models.yaml` (both already unused in GreenServ) | `1d5b6c9` |
| A9 | Removed `--dedicated-queue` and `queue_name` from the telemetry; the orchestrator derives the queue from the node id | `55fc2c8` |
| B1 | `T_max` is a required argument of `Orchestrator`; `run_api.py` reads `max_latency` without a default | `5ed51f7` |
| B2 | `run_api.py` indexes the YAML values directly (except Redis, see B4); the YAML-backed constructor parameters of `Orchestrator` and `AllocationController` have no default | `ebbcfc7` |
| B3 | `load_config` raises on a missing or invalid file and resolves `config/` relative to itself, not the current directory | `4c126b8` |
| B4 | `RedisClient.from_config()` reads the redis block for the API and the worker (flags override it); the constructor has no host/port defaults | `15ba151` |
| B5 | `EvaluationService` holds the reward weights and the formula (`reward`); the orchestrator no longer passes them, and `scripts/placement_prior.py` reuses them | `7c18241` |
| B6 | The `"raw"`/`"exact_match"` defaults only in `_dataset_settings`; known datasets read their keys directly; `task_type` read without a guard. Also corrects the docstring of `scripts/placement_prior.py` from B5 (the live system does not widen its bounds, see A10) | uncommitted |

## A. Dead code

### A6. "Older telemetry" fallback

- **Where:** `_node_free_memory_gb` in `src/services/orchestrator.py:88` (the
  comment "Older telemetry only reports used/total") and its copy in
  `src/services/allocation_controller.py:256`.
- **Problem:** if `gpu_memory_free_gb` is missing, both derive it as total
  minus used. The worker's `_hardware_state` always publishes free, used and
  total together, so this branch can never run.
- **Fix:** delete the branch in both places. C1 then merges the two copies.

### A7. Redundant error handling

- **Where:** the `try/except` around `get_all_node_telemetry()` in
  `process_query` (`orchestrator.py:286`, "Telemetry fetch failed"), and the
  `try/except TypeError` in `RouterService.select_model`
  (`src/services/router_service.py:33`).
- **Problem:** `get_all_node_telemetry` already catches every error and returns
  `[]`. All four selectable bandits accept `available_models`, so the
  `TypeError` branch can only hide a real error and retry without the
  feasibility mask.
- **Fix:** call both directly. The `or []` after the telemetry call can go too.

### A8. Unused parameter and unused config key

- **Where:** `task_type` in `EvaluationService._evaluate`
  (`src/services/evaluation_service.py:21`); `type: "general"` on all 16 models
  in `config/models.yaml`.
- **Problem:** neither is read anywhere in the main code.
- **Fix:** delete both.

### A9. `--dedicated-queue` (needs a decision)

- **Where:** the `dedicated_queue` argument and flag in
  `src/services/inference_service.py` (`:60`, `:98`, `:591`), `queue_name` in
  the telemetry payload (`:442`), and the fallback in `_pick_queue_for_model`
  (`orchestrator.py:242`).
- **Problem:** the flag lets a worker listen on a queue with a custom name, and
  the telemetry carries that name to the orchestrator. If the flag is never
  used, the queue name always follows from the node id.
- **Fix, if unused:** remove the flag and `queue_name` from the telemetry, and
  derive the name from the node id on both sides (see C2).
- **Open question:** is any worker ever started with `--dedicated-queue`?

### A10. `update_bounds` has no effect

- **Where:** the call `self.metric_normalizer.update_bounds(...)` in
  `EvaluationService._normalize_metric` (`src/services/evaluation_service.py`),
  and `MetricNormalizer.update_bounds` in `src/metrics/normalizer.py`. Both
  are GreenServ's (its `evaluation_service.py` already makes the call).
- **Problem:** `update_bounds` stores a widened range under a top-level key
  like `"accuracy_gsm8k"`, but `normalize` reads `bounds["gsm8k"]["accuracy"]`.
  It never sees what `update_bounds` writes, so the live system always uses
  the fixed bounds of `config/metrics.yaml`, clamped to [0, 1]. The thesis
  says exactly that (`model.tex:53`), so the code only looks as if it adapted
  its bounds online.
- **Fix:** remove the call in `_normalize_metric`. `src/metrics/` is out of
  scope, so `update_bounds` itself stays (or goes, if you decide otherwise).
- **Text:** none; the thesis already describes fixed bounds.
- Found on 2026-11-15 while doing B6.

## B. One value, several defaults

### B1. `T_max` has a stale default

- **Where:** `T_max=60.0` in `Orchestrator.__init__` (`orchestrator.py:24`) and
  the fallback `feasibility.get("max_latency", 60.0)` in `run_api.py:61`.
- **Problem:** `config/feature_extraction.yaml` says 30, and so does the
  thesis table. The two 60s are never used but contradict both.
- **Fix:** covered by B2 for `run_api.py`; set or drop the constructor default.

### B2. `run_api.py` repeats YAML values as `.get` defaults

- **Where:** the `.get(key, default)` calls in `run_api.py` on lines 41–107
  (Redis host and port, deadline, load speed, policy, demand, prior, prior
  rate, planning horizon, cooldown).
- **Problem:** every value then exists twice or three times: in the YAML, as
  the `.get` default, and often as a constructor default.
- **Fix:** index directly, as already done with `mab["algorithm"]`, so that a
  missing key fails at startup.

### B3. `load_config` hides missing files

- **Where:** `config/manager.py:13`.
- **Problem:** it returns `{}` on any error. Together with the defaults of B2,
  a mistyped file name silently runs with policy `dynamic`.
- **Fix:** let it raise.
- **Note:** this is GreenServ's file, and `data/` uses it too.

### B4. Redis settings are built twice

- **Where:** `run_api.py:41` and `main()` in `inference_service.py:575`;
  `RedisClient.__init__` has a third default port, 6379
  (`src/queue/redis_client.py:12`), while the config and both callers use 6380.
- **Fix:** one `RedisClient.from_config()` that both call. The worker's
  `--redis-*` flags would become overrides passed to it.

### B5. Reward weights 0.5/0.5 in three places

- **Where:** `Orchestrator.__init__` (`orchestrator.py:22`),
  `EvaluationService.evaluate_and_calculate_reward`
  (`evaluation_service.py:80`), and `scripts/placement_prior.py:26`, which also
  re-implements the reward formula (`:73`).
- **Fix:** `EvaluationService` takes `alpha` and `beta` in its constructor and
  offers the formula as one function; the orchestrator stops passing them and
  the script reuses the function.
- **Text to update:** the `alpha` docstring in `src/bandit/linucb.py` and the
  "Where the parameters are" section of `CLAUDE.md` both say the orchestrator
  holds the reward weights.

### B6. `"raw"` and `"exact_match"` defaults in four places

- **Where:** `_dataset_settings` (`orchestrator.py:76`, `:81`), the call to the
  evaluation (`orchestrator.py:370`), `_evaluate` (`evaluation_service.py:22`)
  and `evaluate_and_calculate_reward` (`evaluation_service.py:78`).
- **Fix:** keep the defaults in one place, so that `_dataset_settings` returns
  them for an unknown dataset too, and remove the others.

### B7. `min_free_gpu_memory_gb` is set twice

- **Where:** `Orchestrator.__init__` (`orchestrator.py:25`) and
  `AllocationController.__init__` (`allocation_controller.py:211`), both 0.5.
- **Problem:** `run_api.py` passes neither, so the filter and the controller
  agree only because the two defaults happen to match.
- **Fix:** pass one value to both, as is done for `load_bandwidth_gb_per_s`.

### B8. Two defaults for the worker's energy method

- **Where:** `energy_method="pynvml"` in `InferenceService.__init__`
  (`inference_service.py:58`) and `default="zeus"` for `--energy-method`
  (`:530`).
- **Fix:** one default. Decide which the thesis uses.

### B9. Thesis-table parameters hidden in method bodies

- **Where:** `MIN_SAMPLES`, `EW` and the 10 % floor in
  `_update_latency_filters` (`allocation_controller.py:323`, `:342`);
  `self.reconcile_interval = 1.0` and `self._load_window = 5` in
  `InferenceService.__init__` (`inference_service.py:88`, `:115`).
- **Problem:** the thesis table lists all five, and `CLAUDE.md` says such
  values are constructor defaults or command-line flags. These are neither.
- **Fix:** make them constructor defaults like the rest.
- **Optional:** `keep_bonus` and the solver time limit in `solve_placement`
  (`allocation_controller.py:111`, `:161`) are also local constants; they are
  not in the thesis table.

## C. Duplicated logic

### C1. Telemetry readers exist twice

- **Where:** `_node_free_memory_gb` and `_avg_latency_for_model` in
  `orchestrator.py:88`, `:99`; `_node_free_memory_gb` and `_latency_for_pair`
  in `allocation_controller.py:256`, `:266`. The controller also applies the
  nominal latency for pairs without history in two places (`:306`, `:423`).
- **Fix:** move the readers next to the other telemetry readers in
  `src/services/model_memory.py`, and rename that module, since it then holds
  more than memory (for example `node_telemetry.py`).
- **Text to update:** the README's "New" entry for `model_memory.py`.

### C2. Queue names are string literals on both sides

- **Where:** `llm_tasks_node_<id>` and `llm_inference_tasks_<tier>` in
  `orchestrator.py:242`, `:322` and `inference_service.py:98`, `:103`.
- **Fix:** two small functions in `redis_client.py` that both sides call.

### C3. Rolling windows in the worker

- **Where:** the get-or-create of a `deque` is written out twice
  (`inference_service.py:236`, `:304`) and the averaging twice (`:381`, `:407`).
- **Fix:** a `defaultdict` of deques and one averaging helper.

### C4. `_pick_queue_for_model`

- **Where:** `orchestrator.py:218`.
- **Problem:** a hand-written weighted random choice in three branches. It also
  recomputes the node options that `_available_models` (`:210`) computed for
  every model, and calls `_traffic_shares` a second time.
- **Fix:** `random.choices`, and compute the options per model once per query.
- **Note:** the `has_plan()` check in `_traffic_shares` (`:168`) is redundant,
  because the controller's shares are empty before its first plan.

### C5. `_get_context_vector`

- **Where:** `src/services/feature_service.py:68`.
- **Problem:** three loops, each behind an `if` that is always true. If a key
  were ever missing, the vector would silently be shorter than
  `context_dimension`.
- **Fix:** one `_one_hot(index, size)` helper, used three times.

### C6. Latency-deviation state in three parallel dicts

- **Where:** `_lat_kalman`, `_lat_resvar`, `_lat_samples`
  (`allocation_controller.py:242`), all keyed by (model, node) and updated
  together in `_update_latency_filters` (`:317`).
- **Fix:** one small class per pair with an `update(observation)` method that
  reports a deviation; the method becomes a short loop.

## D. Structure

### D1. `process_query` does six things

- **Where:** `orchestrator.py:269`, about 110 lines.
- **Fix:** one private helper per step, so the method reads like the step list
  in `CLAUDE.md`:

  ```python
  nodes = self._feasible_nodes()
  features = self.feature_service.extract_features(query_text, metadata)
  model_id, preferred = self._choose_model(features, nodes, model_id)
  task_id = self._dispatch(model_id, nodes, query_text, settings)
  result = await self._wait_for_result(task_id)
  return self._evaluate_and_update(result, features, model_id, reference)
  ```

- **Tests:** they call `process_query`, `_node_options_for_model` (8 times) and
  `_pick_queue_for_model` (once).

### D2. Orchestrator and controller need each other

- **Where:** `run_api.py:75` sets `orch.allocation_controller` after
  construction, and `run_api.py:101` hands the controller one of the
  orchestrator's `consume_*_window` methods.
- **Fix:** move the demand counters (`_windows`, `_count`, `_consume_window`,
  `consume_rps_window`, `consume_shadow_window` in `orchestrator.py:63`,
  `:244`–`:262`) into a small class that both receive. The construction order
  in `run_api.py` then becomes linear, with the controller built before the
  orchestrator.
- **Tests:** they call `consume_rps_window` (once) and `consume_shadow_window`
  (twice).

### D3. Globals and an import inside a function

- **Where:** the module globals in `run_api.py:28` and
  `from run_api import orchestrator` inside `get_orchestrator`
  (`src/api/router.py:30`).
- **Fix:** keep the orchestrator and the controller on `app.state`, set in
  `lifespan` and read in `router.py` and `/health`.
- **Bug in the same file:** the endpoint raises an `HTTPException` inside its
  own `try` (`router.py:52`) and catches it again (`:55`), so a client sees
  `"500: Timeout waiting for response"` as the detail. Reproduced.
- **Note:** `src/api/router.py` is currently identical to GreenServ's.

### D4. The `feasibility` block sits in `feature_extraction.yaml`

- **Where:** `config/feature_extraction.yaml:41`, read in `run_api.py:60`.
- **Problem:** it holds the latency deadline and the load speed, which have
  nothing to do with feature extraction.
- **Fix:** move it to its own file, then flatten `feature_extraction.yaml`
  (its `feature_extraction:` root is the last wrapped root; it is read in
  `feature_service.py:15`). Rename `max_latency` to `max_latency_s`, like the
  other keys with a unit.
- **Text to update:** `CLAUDE.md` describes the `feasibility` block under
  "Where the parameters are". `experiments/config/feature_extraction.yaml` is
  GreenServ's own copy and stays as it is.

## E. Naming and comments

### E1. "Allocation" in the code, "placement" in the thesis

- **Where:** `src/services/allocation_controller.py`, the class
  `AllocationController`, `config/allocation.yaml`, and about 58 other uses in
  code and config.
- **Problem:** the thesis says "placement controller" 13 times and never
  "allocation controller". `CLAUDE.md` needs a sentence to explain the
  mismatch.
- **Fix:** rename the file, the class and the config file.
- **Text to update:** `README.md`, `CLAUDE.md`, a comment in
  `tex/sources/evaluation.tex` that names `config/allocation.yaml`, and the
  tests (13 references).

### E2. One concept under three names

- **Where:** `traffic_shares()`, `weights` and `_weights` in
  `allocation_controller.py` and `orchestrator.py`; `consume_rps_window` next
  to `consume_shadow_window` (`orchestrator.py:257`), where the config says
  `realized`.
- **Fix:** use "shares" throughout and `consume_realized_window`. Storing the
  shares as `{model: {node: share}}` also removes two conversions between
  lists of pairs and dicts.
- **Note:** if D2 is done first, the two `consume_*` methods move anyway.

### E3. Path comments on line 1

- **Where:** fifteen files start with a comment naming their own path.
- **Problem:** four are wrong: `# src/worker.py` in `inference_service.py`,
  `# src/config/manager.py` in `config/manager.py`, `# src/bandit/thompson.py`
  in `thompson_sampling.py`, and `# config/features.yaml` in
  `feature_extraction.yaml`. The other eleven are right but say nothing the
  file name does not.
- **Fix:** remove them, or at least the four wrong ones.

### E4. Stale comments

- **Where and problem:**
  - `orchestrator.py:317` says "Pick the lowest-latency surviving node",
    although a plan with replicas uses the traffic shares.
  - `feature_service.py:51` logs "Setting task_type to None" at warning level
    for every query without a dataset, and then classifies the query anyway.
    This message is GreenServ's.
  - `inference_service.py:9` is commented-out code.
- **Fix:** correct the first, reword and lower the level of the second, delete
  the third.

### E5. Logging

- **Where:** `logging.basicConfig` in `run_api.py:24`, `src/api/router.py:8`
  and `inference_service.py:23`; logger names `"run_api"`, `"api"` and
  `"worker"` next to `__name__` elsewhere; four `print()` calls in
  `config/manager.py`, `redis_client.py` and `feature_service.py`.
- **Fix:** `basicConfig` only in the two entry points, `getLogger(__name__)`
  everywhere, and the prints as log messages.

### E6. README lists

- **Problem:** the three lists at the top of `README.md` do not name four
  files the fork changed: `config/mab.yaml`, `config/feature_extraction.yaml`,
  `config/manager.py` and `src/services/feature_service.py`.
- **Fix:** add them under "Extended".

## Suggested order

1. A6–A10, B1, C3–C5, E3, E4: mechanical, no change in behaviour.
2. B2–B9: these change what happens on a bad setup (it fails at startup
   instead of running with defaults).
3. C1, C2, C6, D1–D4: the real refactors; D1 and D2 touch the tests.
4. E1, E2 last: pure renames that touch the tests, `README.md` and `CLAUDE.md`.

None of the points changes a value in the thesis table.

## Also noticed, not in the list above

- `src/models/registry.py` (an unused `ModelRegistry` class) and
  `src/api/schemas.py` (an empty file) are GreenServ leftovers that nothing
  imports.
- The task classifier is never trained in the live system:
  `config/feature_extraction.yaml` sets `classifier_path:
  "models/task_classifier.pkl"`, but only `models/exemplary_task_classifier.pkl`
  exists. `TaskTypeExtractor` then creates an unfitted `LogisticRegression`,
  whose `predict` raises for every query without a known dataset, and falls
  back to keyword classification. The path is also relative to the current
  directory. Found on 2026-11-15 while doing B3.
- `AllocationController._tick` is declared `async` but never awaits, so a
  placement solve (time limit 10 s) blocks the API's event loop while it runs.
