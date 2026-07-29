test_that("regression matches the pure-R reference on a small example", {
  source(testthat::test_path("../reference", "reference_gbm.R"))
  set.seed(1)
  x <- cbind(x1 = c(1, 2, 3, 4, NA, 6), x2 = c(2, 1, 2, 4, 5, 6))
  y <- c(1, 1.5, 2.2, 3.8, 4.1, 5.9)
  ref <- reference_gbm_fit(x, y, ntrees = 3L, learning_rate = 0.1, max_depth = 2L, min_node_size = 2L)
  fit <- fastgbm(x, y, objective = "reg:squarederror", ntrees = 3L, learning_rate = 0.1, max_depth = 2L, min_node_size = 2L, seed = 1L, verbose = FALSE)
  expect_equal(as.numeric(predict(fit, x, type = "link")), as.numeric(ref$fitted), tolerance = 1e-6)
})

test_that("binary classification returns probabilities", {
  x <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 1)
  y <- c(0, 0, 0, 1, 1, 1)
  fit <- fastgbm(x, y, objective = "binary:logistic", ntrees = 5L, learning_rate = 0.2, max_depth = 2L, min_node_size = 1L, min_child_weight = 0.1, seed = 2L, verbose = FALSE)
  p <- predict(fit, x, type = "response")
  expect_true(all(p >= 0 & p <= 1))
  expect_gte(accuracy(y, p), 0.5)
})

test_that("serialization round trips", {
  x <- matrix(c(1, 2, 3, 4), ncol = 1)
  y <- c(1, 2, 3, 4)
  fit <- fastgbm(x, y, objective = "reg:squarederror", ntrees = 2L, learning_rate = 0.2, max_depth = 1L, min_node_size = 1L, seed = 3L, verbose = FALSE)
  tmp <- tempfile(fileext = ".rds")
  save_fastgbm(fit, tmp)
  fit2 <- load_fastgbm(tmp)
  expect_equal(predict(fit, x), predict(fit2, x))
})

test_that("formula interface expands factors", {
  dat <- data.frame(y = c(0, 0, 1, 1), x1 = c(1, 2, 3, 4), x2 = factor(c("a", "a", "b", "b")))
  fit <- fastgbm(y ~ ., data = dat, objective = "binary:logistic", ntrees = 3L, learning_rate = 0.2, max_depth = 2L, min_node_size = 1L, seed = 4L, verbose = FALSE)
  p <- predict(fit, dat, type = "response")
  expect_length(p, nrow(dat))
})

test_that("cox survival objective trains and predicts survival curves", {
  skip_if_not_installed("survival")
  dat <- na.omit(as.data.frame(survival::lung[, c("time", "status", "age", "sex", "ph.ecog")]))
  x <- model.matrix(~ age + sex + ph.ecog - 1, dat)
  fit <- fastgbm(x, time = dat$time, status = dat$status, objective = "survival:cox", ntrees = 5L, learning_rate = 0.1, max_depth = 2L, min_node_size = 5L, seed = 5L, verbose = FALSE)
  lp <- predict(fit, x, type = "link")
  s <- predict(fit, x[1:8, , drop = FALSE], type = "survival", times = c(30, 90, 365))
  expect_length(lp, nrow(x))
  expect_equal(dim(s), c(8L, 3L))
  expect_true(all(s >= 0 & s <= 1))
  expect_gt(metrics(fit, y = survival::Surv(dat$time, dat$status))$value, 0.5)
})

test_that("aft survival objective trains and predicts survival curves", {
  skip_if_not_installed("survival")
  set.seed(2)
  n <- 80
  x <- matrix(rnorm(n * 3), ncol = 3)
  time <- exp(1 + 0.5 * x[, 1] - 0.2 * x[, 2] + rnorm(n, sd = 0.25))
  status <- rbinom(n, 1, 0.7)
  fit <- fastgbm(x, time = time, status = status, objective = "survival:aft", ntrees = 5L, learning_rate = 0.1, max_depth = 2L, min_node_size = 5L, seed = 6L, verbose = FALSE)
  lp <- predict(fit, x, type = "link")
  s <- predict(fit, x[1:6, , drop = FALSE], type = "survival", times = c(1, 2, 3))
  expect_length(lp, nrow(x))
  expect_equal(dim(s), c(6L, 3L))
  expect_true(all(s >= 0 & s <= 1))
  expect_gt(metrics(fit, y = survival::Surv(time, status))$value, 0.4)
})
