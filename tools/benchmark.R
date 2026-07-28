library(fastgbm)

set.seed(1)
x <- matrix(rnorm(1000), ncol = 5)
y <- x[, 1] * 0.5 - x[, 2] * 0.25 + rnorm(nrow(x), sd = 0.1)
fit <- fastgbm(x, y, objective = "reg:squarederror", ntrees = 20L, verbose = FALSE)
print(summary(fit))
