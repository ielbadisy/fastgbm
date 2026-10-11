## Update submission (0.6.1 -> 0.7.0)

This release adds competing risks to `fastgbm()` through a new `comrisk`
argument (Fine-Gray subdistribution hazard, cause-specific hazards, and
coupled cause-specific hazards, following 'CoxBoost'; Binder et al. 2009
<doi:10.1093/bioinformatics/btp088>), with a new `predict()` method for
cumulative incidence. Existing behaviour and arguments are unchanged.
'CoxBoost' is added to Suggests and is used only in tests, behind
`skip_if_not_installed()`.

## Test environments

* local: Ubuntu 24.04, R 4.5.1 (via `R CMD check --as-cran`)

## R CMD check results

0 errors | 0 warnings | 1 note (local `R CMD check --as-cran`):

* "Compilation used the following non-portable flag(s): '-mno-omit-leaf-frame-pointer'"
  -- this flag comes from the local Ubuntu R build's own `Makeconf`
  (`-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer`, a debug-build
  default), not from this package's `src/Makevars`, which sets only
  `CXX_STD`, `PKG_CXXFLAGS`/`PKG_LIBS` (both via `RcppParallel::CxxFlags()`/
  `RcppParallelLibs()`). Not expected to reproduce on CRAN's own build
  machines.

## Downstream dependencies

There are currently no reverse dependencies on CRAN.
