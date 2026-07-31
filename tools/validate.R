library(fastgbm)

set.seed(1)
n <- 200
x <- matrix(rnorm(n * 4), ncol = 4)
time <- exp(x[, 1] + rnorm(n, sd = 0.2))
status <- rbinom(n, 1, 0.8)
fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 10L, verbose = FALSE)
stopifnot(is.numeric(predict(fit, x)))
