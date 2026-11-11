## Resubmission

This is a resubmission of 0.7.1. The incoming pretest on Debian gave a NOTE:
"Running R code in 'testthat.R' had CPU time 4.4 times elapsed time". The
tests now limit RcppParallel to 2 threads (`tests/testthat.R`) and the
thread-determinism tests compare 1 against 2 threads instead of 4. Locally,
with `_R_CHECK_TEST_TIMING_CPU_TO_ELAPSED_THRESHOLD_=2.5`, the ratio is 1.1
and the NOTE no longer appears. The package code is unchanged.

## Update submission (0.7.0 -> 0.7.1): bug fix

This is an early update, 0.7.0 having been published on 2026-11-03. It fixes
a bug that makes the predictions of `objective = "pexp"` wrong: the time
feature used in training was the midpoint of the time each subject spent in
an interval, whereas prediction uses the midpoint of the whole interval. With
many trees the predicted hazard shrank towards zero and survival curves were
far too high (mean 5-year survival 0.93 against a Kaplan-Meier estimate of
0.44 on the Rotterdam breast cancer data). Users of 0.7.0 get miscalibrated
survival probabilities, so we would like to replace it now rather than wait.
No other change; the API is unchanged.

## Test environments

* local: Ubuntu 24.04, R 4.5.1 (via `R CMD check --as-cran`)

## R CMD check results

0 errors | 0 warnings | 2 notes (local `R CMD check --as-cran`):

* "checking for future file timestamps ... unable to verify current time"
  -- the local machine could not reach the time server; unrelated to the
  package.

* "Compilation used the following non-portable flag(s): '-mno-omit-leaf-frame-pointer'"
  -- this flag comes from the local Ubuntu R build's own `Makeconf`
  (`-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer`, a debug-build
  default), not from this package's `src/Makevars`, which sets only
  `CXX_STD`, `PKG_CXXFLAGS`/`PKG_LIBS` (both via `RcppParallel::CxxFlags()`/
  `RcppParallelLibs()`). Not expected to reproduce on CRAN's own build
  machines.

## Downstream dependencies

There are 2 reverse dependencies on CRAN (funcml, mimar). Both were checked
against 0.7.1 with `R CMD check`: no new problems.
