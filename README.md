# survgbm

`survgbm` is a compact, survival-analysis-only gradient boosting engine for R, with a
compiled (Rcpp + RcppParallel) backend. It targets a specific gap in the R ecosystem:
`xgboost` has a Cox objective but no baseline-hazard/survival-curve prediction built
in, and general-purpose GBM packages don't route missingness natively for
survival-specific workflows. `survgbm` gives you a single function call that handles
Cox, AFT, or piecewise-exponential training, native missing-value routing, baseline
hazard estimation, and survival-probability prediction at chosen horizons.

## Current scope

* Cox (Breslow ties), AFT (normal location-scale), and piecewise-exponential
  objectives - `objective = "cox"` / `"aft"` / `"pexp"`. `pexp` models the hazard
  jointly over covariates *and* time via a person-time expansion (the standard
  Poisson trick), rather than assuming a fixed post-hoc baseline like Cox's Breslow
  estimator - see `vignette("survival", package = "survgbm")`;
* matrix and formula (`Surv(time, status) ~ .`) interfaces;
* histogram-based tree growth with native missing-value routing;
* validation-based early stopping (`validation = list(x=, time=, status=)`,
  `early_stopping = <patience>`) - stops when the held-out loss stops improving and
  keeps the best-iteration model; see Scientific validity below for why this matters;
* deterministic multi-threaded training (`threads = 0L` for automatic; identical
  fitted values regardless of thread count, including with early stopping active);
* baseline hazard estimation and survival-probability prediction at chosen horizons;
* feature importance, partial dependence (`pdp()`), and serialization helpers.

This package is survival-only by design, not survival-only as a current limitation of
a broader plan: multiclass, regression, count/Poisson (general), and classification
objectives are permanently out of scope and will not be added.

## Remaining work

**New objectives** (each needs a derived gradient/Hessian, a finite-difference
validation pass, and a documented estimand before any code is written, per this
project's own rule against undocumented new losses):
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
  (`survgbm:::survival_cindex()`) is still $O(n^2)$, fine at the sample sizes
  benchmarked so far but a real bottleneck for larger cohorts.

**Explored, tested, but not adopted** (results were dataset-dependent, not a
universal win - see `paper/survgbm-benchmark.qmd` Roadmap for the numbers):
* An `mtry`-scaled-to-`sqrt(p)/p` `colsample` default (helped on higher-`p`
  datasets, hurt on low-`p` ones).
* A bagged-ensemble-of-`survgbm`-fits wrapper, bootstrap resampling + averaging,
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
R CMD INSTALL survgbm_0.1.0.tar.gz
```

## Example

```r
library(survgbm)
library(survival)

lung_dat <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
fit <- survgbm(
  as.matrix(lung_dat[, c("age", "sex", "ph.ecog")]),
  time = lung_dat$time,
  status = lung_dat$status,
  objective = "cox"
)
predict(fit, lung_dat[1:5, c("age", "sex", "ph.ecog")], type = "survival", times = c(90, 180, 365))

# or, via formula:
fit2 <- survgbm(Surv(time, status) ~ age + sex + ph.ecog, data = lung_dat)
```

See `vignette("getting-started", package = "survgbm")` and
`vignette("survival", package = "survgbm")` for more.

## Benchmark: survgbm (all 3 objectives) vs. gbm, xgboost, ranger (6 biostatlab survival datasets)

A reproducible benchmark (`inst/benchmarks/run-benchmark.R`) compares all three
`survgbm` objectives (`cox`, `aft`, `pexp`) against `gbm`, `xgboost`, and `ranger`
on 6 real survival datasets from the `biostatlab` package. `gbm`/`xgboost`/`ranger`
use a fixed `ntrees = 200`; each `survgbm` arm uses `ntrees = 200` as a ceiling with
validation-based early stopping active (15% of its training fold held out, patience
20 rounds) - error-analysis diagnostics (`inst/benchmarks/error-analysis/`) showed
test C-index consistently peaked well before 200 rounds without it, and that
`colsample`/`subsample` resampled at every tree node (like `ranger`'s `mtry`, not
xgboost-style per-tree sampling) further improves both speed and C-index. All
arms share `max_depth = 5`, `learning_rate = 0.1`, `min_node_size = 10`,
single-threaded, over 10 repeated 70/30 splits; `survgbm` uses its
`subsample = 0.8`/`colsample = 0.8` defaults. Median training time and Harrell's
C-index (`survgbm`'s best-performing objective per dataset shown, all three
reported in the paper):

| dataset (n) | survgbm (best objective) | gbm | xgboost | ranger |
| --- | --- | --- | --- | --- |
| pbc (276) | 0.006s / 0.809 (cox) | 0.028s / 0.804 | 0.049s / 0.819 | 0.123s / 0.820 |
| heart_failure (299) | 0.003s / **0.706** (cox) | 0.019s / 0.681 | 0.028s / 0.682 | 0.072s / 0.728 |
| breast (672) | 0.006s / **0.666** (aft) | 0.031s / 0.648 | 0.038s / 0.648 | 0.222s / 0.697 |
| colon_cancer (888) | 0.155s / 0.629 (pexp) | 0.047s / 0.645 | 0.035s / 0.609 | 0.193s / 0.655 |
| crc_mondaca2020 (471, real missingness) | 0.005s / **0.576** (cox) | 0.038s / 0.558 | 0.042s / 0.553 | 0.148s / 0.612 |
| framingham (5209) | 0.053s / **0.757** (cox, beats ranger's 0.756 on median) | 0.267s / 0.759 | 0.080s / 0.747 | 2.396s / 0.756 |

(bold = survgbm beats both gbm and xgboost on median C-index)

(training seconds / Harrell's C-index, both medians across 10 repeated 70/30 splits)

Honest summary, not a universal-superiority claim: `survgbm`'s `cox` arm is now
5-9x faster to train than `gbm`/`xgboost` and 22-45x faster than `ranger` on every
dataset, and beats both `gbm` and `xgboost` on median C-index on 4-5 of 6 datasets
depending on objective. `ranger`'s discrimination lead narrowed substantially (e.g.
`breast`: gap went from 5.9 points of C-index to 2.0) and, on `framingham`, `cox`'s
median C-index now edges past `ranger`'s (0.757 vs. 0.756) - though not consistently
enough across repeated splits to count as a clear win in the paired comparison, and
`ranger` remains ahead on the other five datasets. `pexp` is markedly slower than
`cox`/`aft` (it evaluates the ensemble once per time interval, not once per row) and
is currently the strongest objective on only one dataset (`colon_cancer`). See
`paper/survgbm-benchmark.qmd` (renders to `survgbm-benchmark.pdf`) for the full
write-up with repeated-split uncertainty, paired win/loss/tie counts, and the
threads=1-vs-N parallelism results; `inst/benchmarks/benchmark-results.csv` /
`benchmark-summary.csv` / `parallel-speedup.csv` for the raw and aggregated numbers;
and `inst/benchmarks/session-info.txt` for reproducibility metadata.

The earlier single-dataset `survival::lung` benchmark is still available via
`tools/benchmark-survival.R` / `inst/benchmarks/survival-cox-lung.csv`.
