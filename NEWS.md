# fastgbm 0.1.0 (unreleased)

## Scaling benchmark extended to p = 1000

* `inst/benchmarks/run-benchmark-scaling.R`'s `P_GRID` now includes `1000`
  (was `{10, 50, 100}`). Confirms and sharpens the earlier `colsample`
  finding: at `n = 1000`, the fastgbm/ranger training-time ratio reaches
  12.04x (slower) at `p = 1000`, vs. 4.45x at `p = 100`. Unlike lower `p`,
  growing `n` does not rescue `fastgbm` at `p = 1000` within the tested
  range -- it is still ~2.8x slower than `ranger` even at `n = 20000`
  (`colsample` tuning matters most exactly in this regime). The
  thread-count sweep, now run at the new largest grid point (`n = 20000,
  p = 1000`), shows *better* multi-threaded speedup than at `p = 100`
  (5.8x vs. 3.0x at 16 threads) since a wider `colsample`-sampled feature
  set gives the level-wise split search more to parallelize per level.

## Multi-threaded scaling: level-wise (breadth-first) tree growth

* Added `inst/benchmarks/run-benchmark-multithreaded.R`: a direct head-to-head
  of `fastgbm`/`ranger`/`xgboost` at matched thread counts (`{1, 4, 12}`,
  `gbm` shown only as a fixed single-threaded reference), at the largest
  n/p grid point from the scaling benchmark. Finding: the single-threaded
  win over `ranger` reported below did not survive multi-threading --
  `ranger` (independent trees split across threads) and `xgboost`
  (histogram build parallelized across rows and features) both scaled
  substantially better than `fastgbm`'s prior per-node split-search
  parallelism, to the point `fastgbm` became the slowest of the three at
  high thread counts despite winning at `threads = 1`.
* Root cause: the previous per-node `parallelFor()` dispatch (one call per
  node, parallelizing across that node's own sampled features) ran out of
  useful work several levels into a tree -- a single node's row/feature
  count shrinks fast with depth, but the fixed cost of a `parallelFor()`
  call does not, so deep, numerous, small nodes got little benefit from
  more threads.
* Fix: `build_tree()` now grows trees breadth-first, one level at a time.
  Every node active at the current depth has its split-search tasks pooled
  into a single combined batch (`LevelSplitFinder`, replacing the previous
  per-node `SplitFinder`) before any of them run, so the parallelized unit
  of work is an entire tree level, not one node -- keeping per-dispatch
  work large even deep in the tree, since each level roughly doubles its
  node count while roughly halving each node's row count (the total
  per-level work stays close to constant).
* This changes tree structure from the prior depth-first version (a
  different, but equally valid, order of drawing each node's `colsample`
  subset), verified rather than assumed safe: the existing leaf-value
  correctness test and a new multi-threaded-vs-single-threaded
  identical-tree check both pass
  (`tests/testthat/test-tree-building.R`), plus the full pre-existing suite
  (244 tests, unchanged pass count).
* Net effect at `n = 20000, p = 100`: `fastgbm`'s own multi-threaded speedup
  improved from ~1.9x/~3.3x (4/12 threads) to ~2.8x/~5.2x. This narrows,
  but does not close, the gap to `ranger`'s ~4.6x/~9.7x and `xgboost`'s
  ~3.2x/~6.0x -- see README's "Multi-threaded head-to-head" and "Level-wise
  tree growth" sections.
* A smaller first attempt -- increasing `parallelFor()`'s `grainSize` from
  `1` to roughly one chunk per thread, on the theory that fewer/larger
  scheduled tasks would cut scheduling overhead -- was tried, measured, and
  found to make things *worse* (e.g. threads=4 at `n=20000, p=100`: ~1.36s
  -> ~1.56s), then reverted. RcppParallel's own work-stealing scheduler at
  `grainSize = 1` already balanced the per-node case better than a fixed
  static chunking did.

## Split-search performance: child G/H inherited from the parent's split, not recomputed

* Every node in `build_tree()`'s `grow()` recomputed its total
  gradient/Hessian sum (`G`/`H`) with a fresh full pass over its own rows,
  even though those exact values were already produced as a byproduct of
  the *parent's* winning split (`GL`/`HL`/`GR`/`HR`, captured in
  `NodeSplit`'s new fields during the `SplitFinder` histogram scan). `G`/`H`
  are now passed down to each child instead, removing a second full O(rows)
  pass per node on top of the split search itself; only the root node (no
  parent split to inherit from) still computes them directly, once.
* Unlike the two changes below, this one is not bit-identical to before: it
  sums the same values in a different order (histogram-bin order inherited
  from the parent vs. row order for a from-scratch sum), a legitimate
  floating-point reassociation that can occasionally tip a near-tied split
  threshold and produce a different-but-equally-valid tree. Verified
  directly rather than assumed correct: a new test
  (`tests/testthat/test-tree-building.R`) independently recomputes each
  leaf's `-G/(H+lambda)` in R from the actual rows reaching it (`subsample`/
  `colsample` fixed at `1` so that set is unambiguous) and checks it against
  the compiled value to `1e-10`, plus a multi-threaded-vs-single-threaded
  identical-tree check; both pass, along with the full pre-existing suite
  (225 tests, unchanged).
* Net effect on the Friedman1 benchmark (same setup as below): `fastgbm`
  training time dropped further, from ~147ms to ~143-146ms, turning the
  prior statistical tie with `ranger` into a narrow but real win (median
  training time 0.099s vs. `ranger`'s 0.104s over 10 repeated splits,
  win/loss/tie 6/3/1; see `inst/benchmarks/run-benchmark-regression.R`).

## Scaling benchmark: n x p x threads, plus tuning tips

* Added `inst/benchmarks/run-benchmark-scaling.R`: sweeps `n` in
  `{1000, 5000, 20000}` and `p` in `{10, 50, 100}` (Friedman1 plus
  pure-noise columns) comparing `fastgbm` vs. `ranger` training time,
  single-threaded, plus a thread-count sweep (1/2/4/all-cores) at the
  largest grid point.
* Finding: `fastgbm`'s relative speed vs. `ranger` improves as `n` grows
  (1.30x slower at `n=1000, p=10` to 0.57x, i.e. faster, at `n=20000,
  p=10`) but degrades as `p` grows (1.30x -> 4.41x slower at `n=1000` as
  `p` goes 10 -> 100). Root cause identified, not just observed: `fastgbm`'s
  default `colsample = 0.8` searches far more features per split than
  `ranger`'s default `mtry` (`p/3` for regression) once `p` is large;
  matching it (`colsample = 1/3`) turns a loss into a win at `n=5000,
  p=100` (0.98s -> 0.50s, vs. `ranger`'s 0.57s). Multi-threading gives
  real but sub-linear speedup (1.8x at 4 threads, 2.3x at 16, on a 16-core
  machine), as expected for within-tree (not across-tree) parallelism.
* See README's "Scaling behavior vs. n, p, and threads" section for the
  full analysis and resulting tuning-tip list.

## Split-search performance: two fixed costs removed from `src/fastgbm.cpp`

* The per-(node, feature) histogram accumulation in `SplitFinder` made two
  full passes over the node's rows - one only to find the highest occupied
  bin, a second to accumulate gradient/hessian sums into an array sized to
  that bin. Since each feature's bin range is already known globally from
  the initial binning pass (`max_bin_by_feature`, computed once in
  `fastgbm_fit_cpp`), the histogram buffer is now sized up front and a
  single pass both accumulates the sums and finds the tightest split-loop
  bound (the highest bin actually present among these specific rows).
* Removed the per-call heap allocation of the two histogram buffers by
  reusing `thread_local` buffers across every (node, feature) call on a
  given thread instead - safe because `parallelFor()` gives each worker
  thread disjoint index ranges, never concurrent access to the same thread's
  buffer.
* `RcppParallel::parallelFor()` was dispatched for large-enough nodes even
  when `threads = 1L` was explicitly requested, paying real thread-pool
  scheduling overhead for work that only ever ran on one thread. A single
  explicit thread request now always takes the direct serial call,
  regardless of node/feature size.
* Net effect on the Friedman1 regression benchmark (`n = 2000`, `p = 10`,
  200 trees, single-threaded, `bench::mark()` median): `fastgbm` training
  time dropped from ~159ms to ~147ms, closing what was previously a clear
  gap to `ranger` (~143-145ms) to a statistical tie (win/loss/tie 5/5 across
  10 repeated splits in `inst/benchmarks/run-benchmark-regression.R`, and
  `fastgbm`'s own median training time is marginally faster than `ranger`'s:
  0.1005s vs. 0.1015s). See README's "Split-search optimizations" section
  and `inst/benchmarks/regression-exectime-bench.png`/`.csv`.
* All changes are behavior-preserving - identical split decisions, verified
  by the full existing test suite including the thread-count-determinism
  and finite-difference gradient/Hessian checks (`test-parallel.R`,
  `test-gradients.R`).

## Regression and binary classification objectives added back (`objective = "regression"` / `"binary"`)

* `fastgbm()` now covers three task types, not just survival: `objective =
  "regression"` (squared error) and `objective = "binary"` (logistic
  classification, 0/1 response or two-level factor) join the existing
  `"cox"`/`"aft"`/`"pexp"` survival objectives. All three task types share
  the same tree-growing engine, histogram binning, missing-value routing,
  and validation-based early-stopping machinery -- the compiled backend
  (`src/fastgbm.cpp`) only needed two new gradient/Hessian pairs
  (`compute_gaussian_grad_hess`, `compute_logistic_grad_hess`) and matching
  loss functions for early stopping, verified against finite differences in
  `tests/testthat/test-gradients.R`.
* `objective` is now optional: it defaults to `"cox"` when `time`/`status`
  or a `survival::Surv` response is supplied, otherwise it is inferred from
  `y` (a 0/1 vector or two-level factor defaults to `"binary"`, anything
  else numeric to `"regression"`).
* The formula interface (`y ~ .`) now supports non-`Surv` responses for
  regression/binary, not just `Surv(time, status) ~ .`.
* `predict()`, `metrics()` (RMSE for regression, log loss for binary),
  `importance()`, and `pdp()` all work unchanged across the new objectives.
* Survival behavior is unchanged by this work -- the change is additive
  (new objectives reuse the `time` C++ argument slot for the plain response,
  exactly like `pexp` already reuses it for exposure), and the full
  pre-existing survival test suite still passes.

## Per-node feature resampling, default hyperparameters, and a 3-objective benchmark

* Changed feature subsampling (`colsample`) from once-per-tree (xgboost's
  `colsample_bytree` semantics) to once-per-node (`ranger`'s `mtry`
  semantics, resampled at every split): `src/fastgbm.cpp`'s `build_tree()`
  now takes `p`/`colsample`/`rng` instead of a precomputed `feature_ids`
  list, and resamples inside `grow()` at each node. Verified to preserve
  thread-count determinism (all parallel-determinism tests still pass).
  Improves both training speed (smaller effective per-node search) and
  C-index versus per-tree sampling on the benchmark datasets.
* **Bug fix** (found while making this change, applies to all objectives):
  the compiled backend read feature names from a nonexistent `"colnames"`
  attribute on the input matrix (base R matrices store column names under
  `dimnames(x)[[2]]`, not a top-level `"colnames"` attribute), so
  `importance()` always showed generic `V1, V2, ...` labels regardless of
  the matrix's actual `colnames()`.
* Changed `fastgbm()`'s default hyperparameters to match what all this
  session's diagnostics actually validated -- they had drifted out of sync
  with the benchmark's own settings: `learning_rate` 0.05 to 0.1, `max_depth`
  6 to 5, `min_node_size` 20 to 10, `ntrees` 500 to 200. Documented
  explicitly in `?fastgbm` that training all `ntrees` rounds **without**
  `validation`/`early_stopping` reliably overfits on small-to-medium survival
  data, per the diagnostics.
* `inst/benchmarks/run-benchmark.R` now benchmarks all three implemented
  objectives (`fastgbm_cox`, `fastgbm_aft`, `fastgbm_pexp`) as separate arms
  against `gbm`/`xgboost`/`ranger`, not just Cox.
* Explored (not formalized into the package): an `mtry`-scaled-to-`sqrt(p)/p`
  default helped on higher-`p` datasets but hurt on low-`p` ones; a
  bagged-ensemble-of-`fastgbm`-fits wrapper (bootstrap resampling + averaging,
  directly borrowing `ranger`'s core variance-reduction mechanism) closed the
  C-index gap to `ranger` on some datasets (e.g. beat it on `pbc`) but not
  others. See Roadmap in `paper/fastgbm-benchmark.qmd`.

## New objective: piecewise exponential (`objective = "pexp"`)

* Added a piecewise-exponential (PEM) objective. Unlike Cox/AFT (one scalar
  score per subject), `pexp` models the log hazard rate jointly over
  covariates *and* time: training data is expanded into person-time rows via
  the standard "Poisson trick" (`fastgbm_expand_person_time()`, `R/pexp.R`),
  with `log(interval midpoint)` appended as an extra feature, and the ensemble
  is fit with an ordinary Poisson-with-offset gradient/Hessian
  (`compute_pexp_grad_hess()`/`compute_pexp_loss()`, `src/fastgbm.cpp`) -- no
  derivative-of-the-ensemble approximation is needed, unlike a Royston-Parmar-
  style spline baseline, because the hazard is directly `exp(pred)` rather
  than a spline whose local slope must be estimated.
* Cutpoints default to `pexp_bins = 10L` quantiles of the observed event
  times (`fastgbm_pexp_cutpoints()`).
* `predict(type = "survival")` evaluates the hazard-over-time surface
  directly at the requested times (piecewise-linear cumulative hazard,
  extrapolated past the last cutpoint by holding the final interval's hazard
  rate constant). `predict(type = "link"/"response")` uses the cumulative
  hazard at the model's full fitted time horizon as a fixed risk score (no
  single "linear predictor" exists the way Cox/AFT have one, since the model
  is a function of time).
* Validated per this project's own rule against undocumented new losses:
  finite-difference gradient/Hessian check (`tests/testthat/test-gradients.R`),
  a Poisson-GLM-on-expanded-data equivalence check against the coefficient
  `coxph` recovers on the same synthetic proportional-hazards data
  (`tests/testthat/test-pexp.R`), and end-to-end tests for missing-value
  handling, early stopping, thread-count determinism, serialization, and the
  formula interface (same coverage bar as Cox/AFT).
* **Bug fix** (found while building `pexp`, applies to all objectives): the
  compiled backend read feature names from a nonexistent `"colnames"`
  attribute on the input matrix (R matrices store column names under
  `dimnames(x)[[2]]`, not a top-level `"colnames"` attribute), so
  `importance()` always showed generic `V1, V2, ...` labels regardless of the
  matrix's actual `colnames()`. Fixed in `src/fastgbm.cpp`.
* RMST-oriented loss and a Royston-Parmar-style objective remain undesigned
  (see Roadmap in `paper/fastgbm-benchmark.qmd`); `pexp` was implemented
  first because it required no such approximation.

## Error-analysis diagnostics and early stopping

* Added `inst/benchmarks/error-analysis/diagnose.R`: learning-curve, discordant-pair,
  and regularization diagnostics run against `ranger` on all 6 benchmark datasets.
  Found that test-set C-index consistently peaked between 10 and 100 boosting rounds
  and then degraded with further training on every dataset (e.g. `pbc`: train C-index
  0.999 at 300 trees, test C-index peaked at 0.768 at 50 trees, down to 0.753 at the
  previous fixed default of 200) -- classic overfitting from having no way to stop
  early, and a secondary finding that ranking errors concentrate disproportionately in
  a few high-leverage subjects.
* **Implemented validation-based early stopping** (`validation`/`early_stopping`
  parameters previously accepted but ignored, per spec section 14): added
  `compute_cox_loss()`/`compute_aft_loss()` to `src/fastgbm.cpp`, extended
  `fastgbm_fit_cpp()` to bin a validation set with the training cuts, track validation
  loss per round, and stop after `early_stopping` rounds without improvement.
  `fit$trees` is truncated to the best-validation-loss iteration by default; the full
  run is preserved in `fit$n_trees_grown`, `fit$validation_history`,
  `fit$best_iteration`, and `fit$stopping_reason`. Verified deterministic across
  thread counts (`tests/testthat/test-early-stopping.R`).
* Changed `colsample`'s default from `1` to `0.8`, based on the diagnostic
  regularization probe showing small, consistently non-negative C-index gains.
* Rebenchmarked all 6 datasets with early stopping + the new `colsample` default:
  C-index improved on every dataset (e.g. `colon_cancer` 0.605 to 0.625), and training
  time dropped 4-10x since far fewer trees are actually grown -- `fastgbm` is now
  faster to train than `xgboost` on all 6 datasets (previously 4 of 6). `ranger`'s
  discrimination lead narrowed but did not close.
* Repeated the regularization probe with early stopping active (it had only been run
  without it before) and found row subsampling gives a further, mostly-positive gain
  on top of `colsample`; changed `subsample`'s default from `1` to `0.8` as well.
  Rebenchmarked again: `fastgbm` now beats both `gbm` and `xgboost` on C-index on 4 of
  6 datasets (`heart_failure` 0.681/0.682 to 0.703, `breast` 0.648/0.648 to 0.677,
  `crc_mondaca2020` 0.558/0.553 to 0.568, and effectively ties `gbm` on `pbc`); the
  `ranger` gap narrowed further (`breast`: 5.9 points of C-index to 2.0) but did not
  close on any dataset.

## Renamed from fastgbm; narrowed to survival-only

* Package renamed `fastgbm` → `fastgbm`. All identifiers, exported functions, and the
  compiled backend (`src/fastgbm.cpp`) were renamed to match.
* Removed `reg:squarederror` and `binary:logistic` objectives entirely: `fastgbm` is
  now survival-only. Objective strings simplified from `"survival:cox"`/
  `"survival:aft"` to `"cox"`/`"aft"` (default `"cox"`). The plain-`y`-vector
  matrix/formula interface is gone; the response is always `time`/`status` or a
  `survival::Surv` object.
* Removed `rmse()`, `logloss()`, `accuracy()`, `auroc()`, and the pure-R
  squared-error/logistic reference implementation (`tests/reference/`) - no longer
  applicable now that the package only supports survival objectives.
* Added `RcppParallel` as a compiled dependency and parallelized the per-feature
  split search (`SplitFinder` worker in `src/fastgbm.cpp`): each feature's best split
  is computed independently and reduced via a fixed-order argmax, so training is
  bit-identical regardless of thread count. The existing `threads` parameter (0 =
  automatic) is now wired through to `RcppParallel::parallelFor()`; below an
  internal node/feature-count threshold, the same worker code runs serially instead
  (dispatch overhead isn't worth it for small nodes). See `tests/testthat/test-parallel.R`.
* Benchmark expanded from 2 to 6 survival datasets (`pbc`, `heart_failure`,
  `breast`, `colon_cancer`, `crc_mondaca2020`, `framingham`) and the non-survival
  benchmark arms (`pima_diabetes`, `diabetes_prediction`, `maternal_bangladesh`)
  were removed from `inst/benchmarks/run-benchmark.R`.
* NAMESPACE and all `man/*.Rd` files are now fully roxygen2-generated (previously
  hand-written) - run `devtools::document()` after any R-level API change, never
  edit `NAMESPACE` by hand.

## Scientific-validity audit

* **Bug fix**: the AFT censored-observation gradient and Hessian had the wrong
  sign (`grad = pdf/(sigma*sf)` instead of `-pdf/(sigma*sf)`), which pushed
  boosting updates in the wrong direction for every censored AFT observation.
  Found via finite-difference tests against the negative log-likelihood.
* **Bug fix**: `fastgbm_survival_baseline()` (the Breslow baseline cumulative
  hazard used by `predict(..., type = "survival")` for Cox models) computed a
  risk set that only grew and never shrank over time, so the risk-set
  denominator was effectively constant at the total sample size for every
  event time instead of the correct shrinking at-risk set. This distorted
  every Cox survival-probability prediction. Found by comparing against
  `survival::basehaz()` on synthetic data; now matches exactly.
* **Bug fix**: `metrics()` computed the survival C-index using a
  risk-direction score for AFT models that was actually a predicted-time
  score (higher = longer survival, the opposite of "higher = more risk"),
  silently producing near-inverted concordance for AFT models. Fixed by
  negating the AFT linear predictor before scoring.
* **Bug fix**: `importance()` crashed (or silently returned garbage
  data-frame shapes) because the compiled backend never attached feature
  names to `feature_importance`; it now falls back to the model's
  `feature_names`.
* **Bug fix**: `fastgbm()` did not validate that `nrow(x)` matched
  `length(y)` (or `length(time)`/`length(status)`), leading to out-of-bounds
  memory reads and silently garbage predictions on mismatched input; this now
  raises a clear error.
* Added `tests/testthat/test-gradients.R` (finite-difference checks for all
  four objectives, via a new internal `fastgbm_grad_hess()` test hook onto
  the compiled kernels), `test-survival-validity.R` (Breslow baseline vs
  `survival::basehaz`, Cox ranking vs `coxph`), `test-missing.R`, and
  `test-safeguards.R`.
* Added a pure-R exact-greedy `binary:logistic` reference implementation
  (`reference_logistic_fit()`) alongside the existing squared-error one.


* Initial package scaffold.
* Added a compiled gradient boosting backend for squared-error regression and binary logistic classification.
* Added formula and matrix interfaces, prediction, importance, metrics, serialization, tests, and reference code.
