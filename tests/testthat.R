library(testthat)
library(fastgbm)

# CRAN allows at most 2 cores during checks; `threads = 0` (automatic) follows
# this RcppParallel setting.
RcppParallel::setThreadOptions(numThreads = 2)

test_check("fastgbm")
