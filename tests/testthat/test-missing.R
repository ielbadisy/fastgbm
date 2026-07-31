## Missing-value stress tests (spec: single missing value, all-missing feature,
## missingness in newdata not seen in training, missingness concentrated by outcome).

test_that("a single missing value does not break training or prediction", {
  set.seed(1)
  n <- 100
  x <- matrix(rnorm(n * 2), ncol = 2)
  time <- exp(x[, 1] + rnorm(n, sd = 0.1))
  status <- rep(1L, n)
  x[3, 1] <- NA
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 10L, verbose = FALSE)
  p <- predict(fit, x)
  expect_length(p, n)
  expect_true(all(is.finite(p)))
})

test_that("an entirely-missing feature is handled without error and is not used for splitting", {
  set.seed(2)
  n <- 100
  x <- cbind(x1 = rnorm(n), x2 = rep(NA_real_, n))
  time <- exp(x[, 1] + rnorm(n, sd = 0.1))
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 10L,
                min_node_size = 2L, verbose = FALSE)
  p <- predict(fit, x)
  expect_true(all(is.finite(p)))
  imp <- importance(fit)
  x2_gain <- imp$gain[imp$feature == "x2"]
  expect_true(length(x2_gain) == 0L || x2_gain == 0)
})

test_that("missing values concentrated by outcome are routed to reduce loss, not arbitrarily", {
  set.seed(3)
  n <- 200
  x1 <- rnorm(n)
  # High x1 -> short survival (high risk); low x1 -> long survival (low risk).
  time <- ifelse(x1 > 0, rexp(n, rate = 5), rexp(n, rate = 0.2))
  status <- rep(1L, n)
  x1_missing <- x1
  # Missingness only occurs for the low-risk (long-survival) group: an informative pattern.
  x1_missing[x1 <= 0] <- NA
  x <- matrix(x1_missing, ncol = 1)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 20L,
                learning_rate = 0.3, min_node_size = 5L, verbose = FALSE)
  risk <- predict(fit, x, type = "link")
  # Rows with missing x1 all belong to the low-risk group; predicted risk should be lower there.
  expect_lt(mean(risk[is.na(x1_missing)]), mean(risk[!is.na(x1_missing)]))
})

test_that("missingness in newdata that was never observed in training is still handled", {
  set.seed(4)
  n <- 100
  x_train <- matrix(rnorm(n * 2), ncol = 2)
  time_train <- exp(x_train[, 1] + rnorm(n, sd = 0.1))
  status_train <- rep(1L, n)
  fit <- fastgbm(x_train, time = time_train, status = status_train, objective = "cox",
                ntrees = 10L, verbose = FALSE)
  x_new <- matrix(rnorm(10 * 2), ncol = 2)
  x_new[1, ] <- NA
  p <- predict(fit, x_new)
  expect_length(p, 10L)
  expect_true(all(is.finite(p)))
})

test_that("all values missing for a row still produces a finite prediction", {
  set.seed(5)
  n <- 80
  x <- matrix(rnorm(n * 3), ncol = 3)
  time <- exp(x[, 1] + rnorm(n, sd = 0.1))
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 10L, verbose = FALSE)
  x_new <- matrix(NA_real_, nrow = 1, ncol = 3)
  p <- predict(fit, x_new)
  expect_true(is.finite(p))
})
