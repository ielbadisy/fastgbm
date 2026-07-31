## Validation-based early stopping (spec section 14). Diagnostics
## (inst/benchmarks/error-analysis/diagnose.R) showed test-set C-index peaks
## early and then degrades as training continues on every benchmark dataset --
## these tests check the fix for that.

make_cox_data <- function(n, seed) {
  set.seed(seed)
  x <- matrix(rnorm(n * 4), ncol = 4)
  lp <- 0.8 * x[, 1] - 0.5 * x[, 2]
  time <- rexp(n, rate = exp(lp) * 0.2)
  status <- rbinom(n, 1, 0.85)
  list(x = x, time = time, status = status)
}

test_that("early stopping actually stops before ntrees given a validation set", {
  tr <- make_cox_data(300, 1)
  va <- make_cox_data(100, 2)

  fit <- fastgbm(tr$x, time = tr$time, status = tr$status, objective = "cox",
                ntrees = 500L, learning_rate = 0.1, max_depth = 4L, min_node_size = 10L,
                validation = list(x = va$x, time = va$time, status = va$status),
                early_stopping = 10L, threads = 1L, seed = 1L, verbose = FALSE)

  expect_lt(fit$n_trees_grown, 500L)
  expect_equal(fit$stopping_reason, "early_stopping")
  expect_gt(fit$best_iteration, 0L)
  expect_lte(fit$best_iteration, fit$n_trees_grown)
  expect_equal(fit$ntrees, fit$best_iteration)
  expect_equal(length(fit$trees), fit$best_iteration)
  expect_length(fit$validation_history, fit$n_trees_grown)
})

test_that("without validation/early_stopping, behavior is unchanged (trains all ntrees)", {
  tr <- make_cox_data(150, 3)
  fit <- fastgbm(tr$x, time = tr$time, status = tr$status, objective = "cox",
                ntrees = 30L, learning_rate = 0.1, max_depth = 3L, min_node_size = 10L,
                threads = 1L, seed = 1L, verbose = FALSE)
  expect_equal(fit$ntrees, 30L)
  expect_equal(fit$stopping_reason, "disabled")
  expect_equal(fit$n_trees_grown, 30L)
})

test_that("validation and early_stopping must be supplied together", {
  tr <- make_cox_data(80, 4)
  va <- make_cox_data(40, 5)
  expect_error(
    fastgbm(tr$x, time = tr$time, status = tr$status, objective = "cox", ntrees = 20L,
           validation = list(x = va$x, time = va$time, status = va$status), verbose = FALSE),
    "together"
  )
  expect_error(
    fastgbm(tr$x, time = tr$time, status = tr$status, objective = "cox", ntrees = 20L,
           early_stopping = 5L, verbose = FALSE),
    "together"
  )
})

test_that("predictions after truncation match manually truncating trees to best_iteration", {
  tr <- make_cox_data(300, 6)
  va <- make_cox_data(100, 7)
  full_fit <- fastgbm(tr$x, time = tr$time, status = tr$status, objective = "cox",
                      ntrees = 500L, learning_rate = 0.1, max_depth = 4L, min_node_size = 10L,
                      validation = list(x = va$x, time = va$time, status = va$status),
                      early_stopping = 10L, threads = 1L, seed = 1L, verbose = FALSE)

  # Refit without early stopping, capped at n_trees_grown, then hand-truncate to
  # best_iteration the same way the diagnostic script does -- should match exactly.
  raw_fit <- fastgbm(tr$x, time = tr$time, status = tr$status, objective = "cox",
                     ntrees = full_fit$n_trees_grown, learning_rate = 0.1, max_depth = 4L,
                     min_node_size = 10L, threads = 1L, seed = 1L, verbose = FALSE)
  raw_fit$trees <- raw_fit$trees[seq_len(full_fit$best_iteration)]

  expect_equal(predict(full_fit, tr$x, type = "link"), predict(raw_fit, tr$x, type = "link"))
})

test_that("early stopping on AFT trains and produces sane output", {
  set.seed(8)
  n <- 200
  x <- matrix(rnorm(n * 3), ncol = 3)
  time <- exp(1 + 0.5 * x[, 1] + rnorm(n, sd = 0.3))
  status <- rbinom(n, 1, 0.8)
  va_n <- 80
  va_x <- matrix(rnorm(va_n * 3), ncol = 3)
  va_time <- exp(1 + 0.5 * va_x[, 1] + rnorm(va_n, sd = 0.3))
  va_status <- rbinom(va_n, 1, 0.8)

  fit <- fastgbm(x, time = time, status = status, objective = "aft",
                ntrees = 200L, learning_rate = 0.1, max_depth = 3L, min_node_size = 10L,
                validation = list(x = va_x, time = va_time, status = va_status),
                early_stopping = 10L, threads = 1L, seed = 1L, verbose = FALSE)
  expect_true(all(is.finite(predict(fit, x, type = "link"))))
  expect_lte(fit$best_iteration, fit$n_trees_grown)
})

test_that("early stopping is deterministic across thread counts", {
  tr <- make_cox_data(400, 9)
  va <- make_cox_data(120, 10)

  fit1 <- fastgbm(tr$x, time = tr$time, status = tr$status, objective = "cox",
                  ntrees = 200L, learning_rate = 0.1, max_depth = 4L, min_node_size = 5L,
                  validation = list(x = va$x, time = va$time, status = va$status),
                  early_stopping = 10L, threads = 1L, seed = 1L, verbose = FALSE)
  fit4 <- fastgbm(tr$x, time = tr$time, status = tr$status, objective = "cox",
                  ntrees = 200L, learning_rate = 0.1, max_depth = 4L, min_node_size = 5L,
                  validation = list(x = va$x, time = va$time, status = va$status),
                  early_stopping = 10L, threads = 4L, seed = 1L, verbose = FALSE)

  expect_equal(fit1$best_iteration, fit4$best_iteration)
  expect_equal(predict(fit1, tr$x, type = "link"), predict(fit4, tr$x, type = "link"))
})
