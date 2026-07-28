library(fastgbm)

set.seed(1)
x <- matrix(rnorm(200), ncol = 4)
y <- x[, 1] + rnorm(nrow(x), sd = 0.2)
fit <- fastgbm(x, y, objective = "reg:squarederror", ntrees = 10L, verbose = FALSE)
stopifnot(is.numeric(predict(fit, x)))
