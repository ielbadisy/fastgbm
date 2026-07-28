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
