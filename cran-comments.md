## Resubmission

This is a resubmission addressing feedback from Uwe Ligges on the first
submission:

* Single-quoted 'RcppParallel' in the `Description` field.
* Added a methods reference to `Description`: Friedman (2001)
  <doi:10.1214/aos/1013203451>, the canonical gradient boosting machine
  reference.

`fastgbm` is a fast histogram gradient boosting engine with a compiled
(Rcpp + 'RcppParallel') backend, covering regression, binary/multiclass
classification, and right-censored survival analysis (Cox, AFT,
piecewise-exponential) with one interface.

## Test environments

* local: Ubuntu 24.04, R 4.5.1 (via `R CMD check --as-cran`)

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
