## `min_node_size` is a minimum child (leaf) size: no leaf of a tree may hold
## fewer training rows, including rows with a missing value.

leaf_sizes <- function(pred) as.vector(table(round(pred, 10)))

test_that("min_node_size bounds the size of every leaf", {
  set.seed(1)
  n <- 200
  x <- matrix(runif(n), ncol = 1, dimnames = list(NULL, "x1"))
  # an isolated extreme group of 5 rows that a free split would carve out
  y <- ifelse(x[, 1] > 0.97, 10, 0) + rnorm(n, sd = 0.01)
  fit <- fastgbm(x, y = y, objective = "regression", ntrees = 1L, max_depth = 1L,
                 min_node_size = 20L, subsample = 1, colsample = 1, learning_rate = 1,
                 min_child_weight = 0, verbose = FALSE, seed = 1L)
  expect_true(all(leaf_sizes(predict(fit, x)) >= 20L))
})

test_that("min_node_size also holds for deeper trees and missing values", {
  set.seed(2)
  n <- 300
  x <- matrix(rnorm(n * 3), n, 3, dimnames = list(NULL, paste0("x", 1:3)))
  y <- x[, 1] + 2 * (x[, 2] > 1.5) + rnorm(n, sd = 0.1)
  x[sample(n, 8), 1] <- NA
  for (m in c(5L, 15L, 40L)) {
    fit <- fastgbm(x, y = y, objective = "regression", ntrees = 1L, max_depth = 4L,
                   min_node_size = m, subsample = 1, colsample = 1, learning_rate = 1,
                   min_child_weight = 0, verbose = FALSE, seed = 1L)
    expect_true(all(leaf_sizes(predict(fit, x)) >= m))
  }
})

test_that("min_node_size bounds the leaves of the shared ccsh tree", {
  set.seed(3)
  n <- 300
  x <- matrix(rnorm(n * 2), n, 2, dimnames = list(NULL, c("x1", "x2")))
  t1 <- rexp(n, 0.1 * exp(1.5 * (x[, 1] > 1.8)))
  t2 <- rexp(n, 0.1)
  cens <- rexp(n, 0.05)
  time <- pmin(t1, t2, cens)
  status <- ifelse(cens < pmin(t1, t2), 0L, ifelse(t1 < t2, 1L, 2L))
  fit <- fastgbm(x, time = time, status = status, comrisk = "ccsh", ntrees = 1L,
                 max_depth = 3L, min_node_size = 25L, subsample = 1, colsample = 1,
                 learning_rate = 1, min_child_weight = 0, verbose = FALSE, seed = 1L)
  lp <- predict(fit, x, type = "link")
  expect_true(all(leaf_sizes(lp[, 1]) >= 25L))
})

test_that("the formula interface does not use (Intercept) as a feature", {
  set.seed(4)
  d <- data.frame(y = rnorm(60), x1 = rnorm(60), g = factor(sample(c("a", "b", "c"), 60, TRUE)))
  fit <- fastgbm(y ~ ., data = d, objective = "regression", ntrees = 5L, verbose = FALSE)
  expect_false("(Intercept)" %in% fit$feature_names)
  expect_equal(fit$feature_names, c("x1", "gb", "gc"))
  expect_length(predict(fit, d[1:5, ]), 5L)
  expect_false("(Intercept)" %in% importance(fit)$feature)

  # a model saved before the fix (with an `(Intercept)` feature) still predicts
  old <- fastgbm(cbind(`(Intercept)` = 1, model.matrix(~ x1 + g, d)[, -1]), y = d$y,
                 objective = "regression", ntrees = 5L, verbose = FALSE)
  old$formula_terms <- fit$formula_terms
  old$xlevels <- fit$xlevels
  expect_length(predict(old, d[1:5, ]), 5L)
})
