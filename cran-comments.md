## Submission

This is the first submission of `fastgbm` to CRAN.

`fastgbm` is a fast histogram gradient boosting engine with a compiled
(Rcpp + RcppParallel) backend, covering regression, binary classification,
and right-censored survival analysis (Cox, AFT, piecewise-exponential) with
one interface.

## Test environments

* local: Ubuntu 24.04, R 4.5.1 (via `R CMD check --as-cran`)
* win-builder / R-hub: to be run before submission

## R CMD check results

0 errors | 0 warnings | 2 notes (local `R CMD check --as-cran`):

* "New submission" -- expected for a first submission.
* "Compilation used the following non-portable flag(s): '-mno-omit-leaf-frame-pointer'"
  -- this flag comes from the local Ubuntu R build's own `Makeconf`
  (`-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer`, a debug-build
  default), not from this package's `src/Makevars`, which sets only
  `CXX_STD`, `PKG_CXXFLAGS`/`PKG_LIBS` (both via `RcppParallel::CxxFlags()`/
  `RcppParallelLibs()`). Not expected to reproduce on CRAN's own build
  machines.

## Downstream dependencies

None (new package).
