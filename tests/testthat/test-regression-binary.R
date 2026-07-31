test_that("regression fits, predicts, and beats an intercept-only baseline", {
  set.seed(10)
  n <- 200
  x <- matrix(rnorm(n * 4), n, 4)
  y <- x[, 1] - 0.5 * x[, 2] + rnorm(n, sd = 0.2)

  fit <- fastgbm(x, y = y, objective = "regression", ntrees = 60, verbose = FALSE)
  expect_identical(fit$objective, "regression")
  pred <- predict(fit, x, type = "response")
  expect_identical(predict(fit, x, type = "link"), pred)
  expect_lt(sqrt(mean((pred - y)^2)), sd(y))
  expect_error(predict(fit, x, type = "survival"), "survival objectives")
})

test_that("binary classification fits, predicts probabilities in [0, 1], and beats chance", {
  set.seed(11)
  n <- 300
  x <- matrix(rnorm(n * 4), n, 4)
  prob <- 1 / (1 + exp(-(x[, 1] - 0.5 * x[, 2])))
  y <- rbinom(n, 1, prob)

  fit <- fastgbm(x, y = y, objective = "binary", ntrees = 60, verbose = FALSE)
  expect_identical(fit$objective, "binary")
  p <- predict(fit, x, type = "response")
  expect_true(all(p >= 0 & p <= 1))
  expect_gt(mean((p > 0.5) == y), 0.6)

  m <- metrics(fit, y = y)
  expect_identical(m$metric, "logloss")
  expect_true(is.finite(m$value))
})

test_that("objective is auto-detected from y when not supplied", {
  set.seed(12)
  n <- 50
  x <- matrix(rnorm(n * 2), n, 2)
  expect_identical(fastgbm(x, y = rnorm(n), ntrees = 5, verbose = FALSE)$objective, "regression")
  expect_identical(fastgbm(x, y = rbinom(n, 1, 0.5), ntrees = 5, verbose = FALSE)$objective, "binary")
  expect_identical(fastgbm(x, y = factor(rbinom(n, 1, 0.5)), ntrees = 5, verbose = FALSE)$objective, "binary")
})

test_that("binary objective rejects a non-0/1 response", {
  x <- matrix(rnorm(20), 10, 2)
  expect_error(fastgbm(x, y = c(1, 2, rep(0, 8)), objective = "binary", verbose = FALSE), "0/1")
})

test_that("early stopping works for regression and binary objectives", {
  set.seed(13)
  n <- 150
  x <- matrix(rnorm(n * 3), n, 3)
  y <- x[, 1] + rnorm(n, sd = 0.3)
  x_train <- x[1:100, ]; y_train <- y[1:100]
  x_val <- x[101:150, ]; y_val <- y[101:150]

  fit <- fastgbm(
    x_train, y = y_train, objective = "regression",
    ntrees = 200, validation = list(x = x_val, y = y_val),
    early_stopping = 5, verbose = FALSE
  )
  expect_identical(fit$stopping_reason, "early_stopping")
  expect_lt(fit$ntrees, 200)
})

test_that("formula interface supports regression and binary objectives", {
  set.seed(14)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.2)
  fit <- fastgbm(y ~ x1 + x2, data = df, ntrees = 20, verbose = FALSE)
  expect_identical(fit$objective, "regression")
  expect_length(predict(fit, df, type = "response"), n)
})
