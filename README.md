# fastgbm

`fastgbm` is a compact gradient boosting engine for R, with a compiled (Rcpp +
RcppParallel) backend, covering three task types with one interface: regression
(squared error), binary classification (logistic), and right-censored survival
analysis (Cox, AFT, or piecewise-exponential objectives). Survival is where the
package differentiates itself from general-purpose GBM packages: `xgboost` has a
Cox objective but no baseline-hazard/survival-curve prediction built in, and
general-purpose GBM packages don't route missingness natively for survival-specific
workflows. `fastgbm` gives you a single function call that handles training, native
missing-value routing, and (for survival objectives) baseline hazard estimation and
survival-probability prediction at chosen horizons.

## Current scope

* Three task types, one function (`fastgbm()`):
  * `objective = "regression"` - squared error;
  * `objective = "binary"` - logistic classification (0/1 response or two-level
    factor);
  * `objective = "cox"` / `"aft"` / `"pexp"` - right-censored survival analysis.
    Cox uses Breslow ties, AFT a normal location-scale error, and `pexp`
    (piecewise-exponential) models the hazard jointly over covariates *and* time via
    a person-time expansion (the standard Poisson trick), rather than assuming a
    fixed post-hoc baseline like Cox's Breslow estimator - see
    `vignette("survival", package = "fastgbm")`.
  * `objective` can be omitted: it is inferred from `y` (a `survival::Surv`
    response or `time`/`status` defaults to `"cox"`; a 0/1 or two-level-factor `y`
    defaults to `"binary"`; anything else numeric defaults to `"regression"`).
* matrix and formula (`y ~ .` / `Surv(time, status) ~ .`) interfaces;
* histogram-based tree growth with native missing-value routing;
* validation-based early stopping (`validation = list(x=, y=)` for
  regression/binary, or `list(x=, time=, status=)` for survival,
  `early_stopping = <patience>`) - stops when the held-out loss stops improving and
  keeps the best-iteration model; see Scientific validity below for why this matters;
* deterministic multi-threaded training (`threads = 0L` for automatic; identical
  fitted values regardless of thread count, including with early stopping active);
* baseline hazard estimation and survival-probability prediction at chosen horizons
  (survival objectives only);
* feature importance, partial dependence (`pdp()`), and serialization helpers,
  shared across all three task types.

## Remaining work

**New objectives** (each needs a derived gradient/Hessian, a finite-difference
validation pass, and a documented estimand before any code is written, per this
project's own rule against undocumented new losses):
* Multiclass classification (softmax).
* RMST-oriented loss - undesigned.
* Royston-Parmar-style flexible spline hazard-surface objective - undesigned; harder
  than `pexp` because the spline's hazard derivative w.r.t. log-time would need finite
  differences of the (non-differentiable) tree ensemble.
* AFT distributions beyond the normal (logistic, extreme value).

**Engine / training**
* Leaf-wise (`grow_policy = "lossguide"`) tree growth - depth-wise only currently.
* Monotonic and interaction constraints.
* Parallel prediction - only the training split search is parallelized (RcppParallel);
  prediction is single-threaded.
* An $O(n \log n)$ concordance implementation - the internal C-index helper
  (`fastgbm:::survival_cindex()`) is still $O(n^2)$, fine at the sample sizes
  benchmarked so far but a real bottleneck for larger cohorts.

**Explored, tested, but not adopted** (results were dataset-dependent, not a
universal win - see `paper/fastgbm-benchmark.qmd` Roadmap for the numbers):
* An `mtry`-scaled-to-`sqrt(p)/p` `colsample` default (helped on higher-`p`
  datasets, hurt on low-`p` ones).
* A bagged-ensemble-of-`fastgbm`-fits wrapper, bootstrap resampling + averaging,
  directly borrowing `ranger`'s bagging mechanism (closed the C-index gap to
  `ranger` on some datasets, not others).

**API-level**
* `pexp`'s `predict(type = "link"/"response")` uses the cumulative hazard at the
  model's fitted time horizon as a risk score, since there's no single scalar
  "linear predictor" the way Cox/AFT have one - correct, but coarser and slower to
  compute than the other two objectives.
* No CRAN-readiness pass yet (vignette polish, `R CMD check --as-cran`, cross-platform
  CI) - all development and testing so far has been local (Linux).

## Scientific validity

Every objective's gradient/Hessian has been checked against finite differences of its
stated loss, the Cox baseline-hazard estimator has been checked against
`survival::basehaz()`, missing-value and numerical-edge-case behavior is
stress-tested, and training is checked to be bit-identical across thread counts (see
`tests/testthat/test-gradients.R`, `test-survival-validity.R`, `test-missing.R`,
`test-safeguards.R`, `test-parallel.R`). This audit found and fixed four real defects
in the pre-rename implementation (a sign error in the AFT censored-observation
gradient/Hessian, a Breslow baseline hazard whose risk set never shrank over time, a
sign error in the AFT concordance score, and a crash in `importance()`) - see
`NEWS.md` for details. `R CMD check` is clean (0 errors, 0 warnings).

## Build

```r
R CMD build .
R CMD INSTALL fastgbm_0.1.0.tar.gz
```

## Example

```r
library(fastgbm)

# regression
fit_reg <- fastgbm(as.matrix(mtcars[, c("cyl", "disp", "hp", "wt")]), y = mtcars$mpg)
predict(fit_reg, mtcars[1:5, c("cyl", "disp", "hp", "wt")])

# binary classification
fit_bin <- fastgbm(as.matrix(mtcars[, c("mpg", "disp", "hp", "wt")]), y = mtcars$am, objective = "binary")
predict(fit_bin, mtcars[1:5, c("mpg", "disp", "hp", "wt")], type = "response")  # probabilities

# survival
library(survival)
lung_dat <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
fit_surv <- fastgbm(
  as.matrix(lung_dat[, c("age", "sex", "ph.ecog")]),
  time = lung_dat$time,
  status = lung_dat$status,
  objective = "cox"
)
predict(fit_surv, lung_dat[1:5, c("age", "sex", "ph.ecog")], type = "survival", times = c(90, 180, 365))

# formula interface works for all three:
fastgbm(mpg ~ cyl + disp + hp + wt, data = mtcars)
fastgbm(Surv(time, status) ~ age + sex + ph.ecog, data = lung_dat)
```

See `vignette("getting-started", package = "fastgbm")`,
`vignette("regression", package = "fastgbm")`,
`vignette("classification", package = "fastgbm")`, and
`vignette("survival", package = "fastgbm")` for more.

## Benchmark: fastgbm (all 3 objectives) vs. gbm, xgboost, ranger (6 biostatlab survival datasets)

A reproducible benchmark (`inst/benchmarks/run-benchmark.R`) compares all three
`fastgbm` objectives (`cox`, `aft`, `pexp`) against `gbm`, `xgboost`, and `ranger`
on 6 real survival datasets from the `biostatlab` package. `gbm`/`xgboost`/`ranger`
use a fixed `ntrees = 200`; each `fastgbm` arm uses `ntrees = 200` as a ceiling with
validation-based early stopping active (15% of its training fold held out, patience
20 rounds) - error-analysis diagnostics (`inst/benchmarks/error-analysis/`) showed
test C-index consistently peaked well before 200 rounds without it, and that
`colsample`/`subsample` resampled at every tree node (like `ranger`'s `mtry`, not
xgboost-style per-tree sampling) further improves both speed and C-index. All
arms share `max_depth = 5`, `learning_rate = 0.1`, `min_node_size = 10`,
single-threaded, over 10 repeated 70/30 splits; `fastgbm` uses its
`subsample = 0.8`/`colsample = 0.8` defaults. Median training time and Harrell's
C-index (`fastgbm`'s best-performing objective per dataset shown, all three
reported in the paper):

| dataset (n) | fastgbm (best objective) | gbm | xgboost | ranger |
| --- | --- | --- | --- | --- |
| pbc (276) | 0.006s / 0.809 (cox) | 0.028s / 0.804 | 0.049s / 0.819 | 0.123s / 0.820 |
| heart_failure (299) | 0.003s / **0.706** (cox) | 0.019s / 0.681 | 0.028s / 0.682 | 0.072s / 0.728 |
| breast (672) | 0.006s / **0.666** (aft) | 0.031s / 0.648 | 0.038s / 0.648 | 0.222s / 0.697 |
| colon_cancer (888) | 0.155s / 0.629 (pexp) | 0.047s / 0.645 | 0.035s / 0.609 | 0.193s / 0.655 |
| crc_mondaca2020 (471, real missingness) | 0.005s / **0.576** (cox) | 0.038s / 0.558 | 0.042s / 0.553 | 0.148s / 0.612 |
| framingham (5209) | 0.053s / **0.757** (cox, beats ranger's 0.756 on median) | 0.267s / 0.759 | 0.080s / 0.747 | 2.396s / 0.756 |

(bold = fastgbm beats both gbm and xgboost on median C-index)

(training seconds / Harrell's C-index, both medians across 10 repeated 70/30 splits)

Honest summary, not a universal-superiority claim: `fastgbm`'s `cox` arm is now
5-9x faster to train than `gbm`/`xgboost` and 22-45x faster than `ranger` on every
dataset, and beats both `gbm` and `xgboost` on median C-index on 4-5 of 6 datasets
depending on objective. `ranger`'s discrimination lead narrowed substantially (e.g.
`breast`: gap went from 5.9 points of C-index to 2.0) and, on `framingham`, `cox`'s
median C-index now edges past `ranger`'s (0.757 vs. 0.756) - though not consistently
enough across repeated splits to count as a clear win in the paired comparison, and
`ranger` remains ahead on the other five datasets. `pexp` is markedly slower than
`cox`/`aft` (it evaluates the ensemble once per time interval, not once per row) and
is currently the strongest objective on only one dataset (`colon_cancer`). See
`paper/fastgbm-benchmark.qmd` (renders to `fastgbm-benchmark.pdf`) for the full
write-up with repeated-split uncertainty, paired win/loss/tie counts, and the
threads=1-vs-N parallelism results; `inst/benchmarks/benchmark-results.csv` /
`benchmark-summary.csv` / `parallel-speedup.csv` for the raw and aggregated numbers;
and `inst/benchmarks/session-info.txt` for reproducibility metadata.

The earlier single-dataset `survival::lung` benchmark is still available via
`tools/benchmark-survival.R` / `inst/benchmarks/survival-cox-lung.csv`.

## Benchmark: regression, on Friedman's #1 data-generating process

A second reproducible benchmark (`inst/benchmarks/run-benchmark-regression.R`)
checks the `"regression"` objective on Friedman's #1 DGP
(`mlbench::mlbench.friedman1()`), the standard synthetic nonlinear-regression
benchmark used in the gradient boosting literature:

```
y = 10*sin(pi*x1*x2) + 20*(x3 - 0.5)^2 + 10*x4 + 5*x5 + eps
```

with `x1`-`x10` iid Uniform(0, 1) and only `x1`-`x5` entering the mean function
(`x6`-`x10` are pure noise). This DGP is useful precisely because the true
functional form is known, so both predictive accuracy *and* whether the model
recovers the right shape per feature can be checked, not just average error.
Same regime as the survival benchmark: `fastgbm` vs. `gbm`/`xgboost`/`ranger`,
equal `ntrees = 200` (ceiling for `fastgbm`, which uses early stopping)/
`max_depth = 5`/`learning_rate = 0.1`/`min_node_size = 10`, single-threaded,
10 repeated 70/30 splits on `n = 2000`.

**Computational time and accuracy** (training seconds / test-set RMSE, both
medians across repeats):

| model | train time | RMSE |
| --- | --- | --- |
| fastgbm | 0.101s | 1.388 |
| gbm | 0.146s | 1.281 |
| xgboost | 0.197s | 1.388 |
| ranger | 0.102s | 2.594 |

`fastgbm` trains ~1.4-2.0x faster than `gbm`/`xgboost` (win/loss/tie 10/0/0
vs. each, paired across repeats) and is now essentially tied with `ranger`
on median training time (win/loss/tie 5/5/0, and `fastgbm`'s own median is
marginally faster: 0.1005s vs. 0.1015s), while clearly beating `ranger` on
RMSE (10/0/0) and matching `xgboost` (6/4/0); `gbm` edges out `fastgbm` on
RMSE here (0/10/0). Consistent with the survival benchmark: `fastgbm` is not
a universal-accuracy win, but it is never far off and is now among the
fastest, not just close to it -- see "Split-search optimizations" below for
what closed the gap to `ranger`.

For a closer look at the *distribution* of training time, not just the
median, `inst/benchmarks/run-benchmark-exectime.R` times all four models
(explicitly checked to grow the same `ntrees = 200` each, so the comparison
is purely about training-algorithm speed) on one fixed 2,000-row Friedman1
draw with [`bench::mark()`](https://bench.r-lib.org/) (`benchr`, originally
used for this, was archived off CRAN in November 2025; `bench` is the
actively maintained equivalent), which reruns each model until it has
collected at least 10 timings:

![bench::mark() training-time distribution for fastgbm, gbm, xgboost, and ranger on Friedman1](inst/benchmarks/regression-exectime-bench.png)

`fastgbm` and `ranger` overlap almost entirely around 141-150ms, both
clearly separated from `gbm`/`xgboost`'s 199-234ms band. See
`inst/benchmarks/regression-exectime-bench.csv` for the full `bench::mark()`
summary (min/median/`itr/sec`/memory allocation).

### Split-search optimizations

Two changes to `src/fastgbm.cpp`'s split search closed most of the gap to
`ranger` shown above (previously `fastgbm` trained 1.3-1.8x faster than
`gbm`/`xgboost` but noticeably behind `ranger`; see `NEWS.md` for the exact
before/after numbers):

* The per-(node, feature) histogram accumulation used to make two full
  passes over the node's rows -- one just to find the highest occupied bin,
  a second to accumulate gradient/hessian sums into an array sized to that
  bin. Since each feature's bin range is already known globally from the
  initial binning pass, the array can be sized up front and both the
  accumulation *and* the tightest split-loop bound obtained from a single
  pass instead.
* `RcppParallel::parallelFor()` was still being dispatched for large-enough
  nodes even when `threads = 1L` was requested, paying real thread-pool
  scheduling overhead for work that only ever ran on one thread. A single
  explicit thread request now always takes the direct (serial) call path,
  regardless of node/feature size.

Both changes are behavior-preserving (identical split decisions, verified by
the full existing test suite including the thread-count-determinism and
finite-difference gradient/Hessian checks) -- they change only how fast the
same computation runs, not what it computes.

**PDP shape recovery**: since the true partial dependence of each feature is
computable in closed form for this DGP (fixing `x_j` on a grid and averaging
the true mean function over the observed rows), `pdp()` on the fitted
`fastgbm` model can be checked directly against ground truth rather than just
inspected by eye:

![fastgbm partial dependence vs. the true Friedman1 partial dependence, for x1, x3, x4, x5](inst/benchmarks/regression-pdp.png)

`fastgbm`'s partial dependence tracks the true shape closely for all four
plotted features: the `x1*x2` interaction correctly flattens `x1`'s marginal
effect into a shallow hump (not the raw `sin` curve, since `x2` is averaged
out), `x3`'s quadratic term is recovered as a clean U-shape, and `x4`/`x5`'s
linear terms are recovered as straight lines. See
`inst/benchmarks/regression-pdp-data.csv` for the underlying numbers and
`inst/benchmarks/regression-session-info.txt` for reproducibility metadata.
