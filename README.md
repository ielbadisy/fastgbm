# fastgbm

`fastgbm` is an R package for compact gradient boosting with a compiled backend.

## Current scope

This initial implementation provides:

* matrix and formula interfaces;
* regression and binary classification objectives;
* survival:cox and survival:aft objectives;
* histogram-based tree growth with missing-value handling;
* deterministic seeding;
* prediction, feature importance, and serialization helpers;
* a pure-R reference implementation for regression tests.

## Build

```r
R CMD build .
R CMD INSTALL fastgbm_0.1.0.tar.gz
```

## Example

```r
library(fastgbm)
fit <- fastgbm(iris[, 1:4], as.numeric(iris$Species == "setosa"), objective = "binary:logistic")
predict(fit, iris[1:5, 1:4], type = "response")
```

## Survival benchmark

On a reproducible `survival::lung` Cox benchmark, the current snapshot produced:

| model | train sec | predict sec | C-index |
| --- | ---: | ---: | ---: |
| fastgbm | 0.003 | 0.000 | 0.421 |
| gbm | 0.005 | 0.001 | 0.397 |
| xgboost | 0.020 | 0.002 | 0.430 |
| ranger | 0.013 | 0.005 | 0.382 |

See `tools/benchmark-survival.R` and `inst/benchmarks/survival-cox-lung.csv`.
