# fastgbm

`fastgbm` is an R package for compact gradient boosting with a compiled backend.

## Current scope

This initial implementation provides:

* matrix and formula interfaces;
* regression and binary classification objectives;
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
