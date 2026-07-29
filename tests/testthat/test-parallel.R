## The feature-level split search runs through RcppParallel when the node/feature
## count crosses an internal threshold. Since each feature's split candidate is
## computed independently and reduced via a fixed-order argmax (src/survgbm.cpp),
## training must be bit-identical regardless of thread count.

test_that("training is deterministic across thread counts", {
  set.seed(1)
  n <- 400
  p <- 12
  x <- matrix(rnorm(n * p), ncol = p)
  lp_true <- rowSums(x[, 1:3] * matrix(c(1, -0.7, 0.5), nrow = n, ncol = 3, byrow = TRUE))
  time <- rexp(n, rate = exp(lp_true) * 0.2)
  status <- rbinom(n, 1, 0.8)

  fit1 <- survgbm(x, time = time, status = status, objective = "cox", ntrees = 20L,
                  learning_rate = 0.1, max_depth = 4L, min_node_size = 5L,
                  threads = 1L, seed = 1L, verbose = FALSE)
  fit4 <- survgbm(x, time = time, status = status, objective = "cox", ntrees = 20L,
                  learning_rate = 0.1, max_depth = 4L, min_node_size = 5L,
                  threads = 4L, seed = 1L, verbose = FALSE)

  expect_equal(predict(fit1, x, type = "link"), predict(fit4, x, type = "link"))
  expect_equal(fit1$feature_importance, fit4$feature_importance)
})
