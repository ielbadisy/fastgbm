library(survgbm)

set.seed(1)
n <- 1000
x <- matrix(rnorm(n * 5), ncol = 5)
time <- exp(rnorm(n))
status <- rbinom(n, 1, 0.8)
Rprof(tmp <- tempfile())
fit <- survgbm(x, time = time, status = status, objective = "cox", ntrees = 20L, verbose = FALSE)
Rprof(NULL)
summaryRprof(tmp)
