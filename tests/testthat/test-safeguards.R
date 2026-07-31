## Numerical-safeguard tests (spec section 7): constant features, degenerate
## nodes, tiny/zero-variance data must never produce non-finite output.

test_that("a constant feature is skipped and does not crash training", {
  set.seed(1)
  n <- 60
  x <- cbind(x1 = rnorm(n), x2 = rep(3, n))
  time <- exp(x[, 1] + rnorm(n, sd = 0.1))
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 10L, verbose = FALSE)
  expect_true(all(is.finite(predict(fit, x))))
  imp <- importance(fit)
  expect_true(all(imp$feature != "x2" | imp$gain == 0))
})

test_that("all-constant predictors still fit without error", {
  set.seed(2)
  n <- 30
  x <- matrix(rep(1, n * 2), ncol = 2)
  time <- exp(rnorm(n))
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 5L, verbose = FALSE)
  p <- predict(fit, x)
  expect_true(all(is.finite(p)))
  # No feature can split, so every tree is a single leaf: identical predictions for all rows.
  expect_equal(length(unique(round(p, 8))), 1L)
})

test_that("a single-row dataset does not crash training or prediction", {
  x <- matrix(c(1, 2), nrow = 1)
  fit <- fastgbm(x, time = 3, status = 1L, objective = "cox", ntrees = 3L,
                min_node_size = 1L, verbose = FALSE)
  expect_true(is.finite(predict(fit, x)))
})

test_that("a tiny two-row dataset with min_node_size forcing no splits still yields finite leaf values", {
  x <- matrix(c(1, 2), ncol = 1)
  time <- c(10, 20)
  status <- c(1L, 1L)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 5L,
                min_node_size = 10L, verbose = FALSE)
  p <- predict(fit, x)
  expect_true(all(is.finite(p)))
  # min_node_size larger than n forces every tree to be a single leaf.
  expect_equal(p[1], p[2], tolerance = 1e-8)
})

test_that("extreme outlier survival times do not produce non-finite predictions", {
  set.seed(3)
  n <- 100
  x <- matrix(rnorm(n), ncol = 1)
  time <- exp(x[, 1])
  time[1] <- 1e8
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 20L,
                learning_rate = 0.1, verbose = FALSE)
  expect_true(all(is.finite(predict(fit, x))))
})

test_that("AFT with near-perfectly separated risk groups stays numerically finite", {
  set.seed(4)
  n <- 100
  x <- matrix(c(rnorm(n / 2, -5), rnorm(n / 2, 5)), ncol = 1)
  time <- exp(x[, 1] * 2)
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "aft", ntrees = 50L,
                learning_rate = 0.3, min_node_size = 2L, verbose = FALSE)
  p <- predict(fit, x, type = "link")
  expect_true(all(is.finite(p)))
})

test_that("zero-event (all-censored) survival data does not crash Cox training", {
  set.seed(5)
  n <- 50
  x <- matrix(rnorm(n), ncol = 1)
  time <- runif(n, 1, 10)
  status <- rep(0L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox",
                ntrees = 5L, min_node_size = 5L, verbose = FALSE)
  p <- predict(fit, x, type = "link")
  expect_true(all(is.finite(p)))
})

test_that("all-event survival data (no censoring) trains and predicts finitely", {
  set.seed(6)
  n <- 50
  x <- matrix(rnorm(n), ncol = 1)
  time <- exp(rnorm(n))
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox",
                ntrees = 10L, min_node_size = 5L, verbose = FALSE)
  p <- predict(fit, x, type = "link")
  expect_true(all(is.finite(p)))
})

test_that("heavily tied event times are handled by the Cox objective", {
  set.seed(7)
  n <- 60
  x <- matrix(rnorm(n), ncol = 1)
  time <- rep(c(1, 2, 3, 4, 5), length.out = n)
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox",
                ntrees = 10L, min_node_size = 5L, verbose = FALSE)
  p <- predict(fit, x, type = "link")
  expect_true(all(is.finite(p)))
})
