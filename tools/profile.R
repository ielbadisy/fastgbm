library(fastgbm)

set.seed(1)
x <- matrix(rnorm(1000), ncol = 5)
y <- rnorm(nrow(x))
Rprof(tmp <- tempfile())
fit <- fastgbm(x, y, objective = "reg:squarederror", ntrees = 20L, verbose = FALSE)
Rprof(NULL)
summaryRprof(tmp)
