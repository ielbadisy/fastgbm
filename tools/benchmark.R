library(survgbm)

set.seed(1)
n <- 1000
x <- matrix(rnorm(n * 5), ncol = 5)
lp_true <- x[, 1] * 0.5 - x[, 2] * 0.25
time <- rexp(n, rate = exp(lp_true) * 0.2)
status <- rbinom(n, 1, 0.8)
fit <- survgbm(x, time = time, status = status, objective = "cox", ntrees = 20L, verbose = FALSE)
print(summary(fit))
