test_that("multiclass is inferred automatically from a factor response with 3+ levels", {
  set.seed(1)
  x <- as.matrix(iris[, 1:4])
  fit <- fastgbm(x, y = iris$Species, ntrees = 20, verbose = FALSE)
  expect_s3_class(fit, "fastgbm_multiclass")
  expect_equal(fit$objective, "multiclass")
  expect_equal(fit$levels, levels(iris$Species))
})

test_that("objective = 'multiclass' can be requested explicitly", {
  set.seed(1)
  x <- as.matrix(iris[, 1:4])
  fit <- fastgbm(x, y = iris$Species, objective = "multiclass", ntrees = 20, verbose = FALSE)
  expect_s3_class(fit, "fastgbm_multiclass")
})

test_that("predict(type = 'prob') returns a row-stochastic matrix with the right dimnames", {
  set.seed(1)
  x <- as.matrix(iris[, 1:4])
  fit <- fastgbm(x, y = iris$Species, ntrees = 20, verbose = FALSE)
  probs <- predict(fit, x, type = "prob")
  expect_equal(dim(probs), c(nrow(x), nlevels(iris$Species)))
  expect_equal(colnames(probs), levels(iris$Species))
  expect_equal(rowSums(probs), rep(1, nrow(x)), tolerance = 1e-10)
  expect_true(all(probs >= 0 & probs <= 1))
})

test_that("predict(type = 'class') is the argmax of predict(type = 'prob')", {
  set.seed(1)
  x <- as.matrix(iris[, 1:4])
  fit <- fastgbm(x, y = iris$Species, ntrees = 20, verbose = FALSE)
  probs <- predict(fit, x, type = "prob")
  cls <- predict(fit, x, type = "class")
  expect_s3_class(cls, "factor")
  expect_equal(levels(cls), levels(iris$Species))
  expect_equal(as.character(cls), colnames(probs)[max.col(probs, ties.method = "first")])
})

test_that("multiclass fit recovers a well-separated synthetic 3-class problem", {
  set.seed(42)
  n <- 300
  cls <- sample(c("a", "b", "c"), n, replace = TRUE)
  mu <- c(a = -3, b = 0, c = 3)
  x <- cbind(x1 = rnorm(n, mu[cls], 0.5), x2 = rnorm(n))
  fit <- fastgbm(x, y = factor(cls), ntrees = 50, verbose = FALSE)
  pred <- predict(fit, x, type = "class")
  expect_gt(mean(as.character(pred) == cls), 0.9)
})

test_that("metrics() reports accuracy and multiclass log loss", {
  set.seed(1)
  x <- as.matrix(iris[, 1:4])
  fit <- fastgbm(x, y = iris$Species, ntrees = 20, verbose = FALSE)
  m <- metrics(fit, y = iris$Species)
  expect_equal(m$objective, "multiclass")
  expect_true(all(c("accuracy", "mlogloss") %in% names(m$value)))
  expect_true(m$value["accuracy"] >= 0 && m$value["accuracy"] <= 1)
  expect_true(m$value["mlogloss"] >= 0)
})

test_that("importance() combines gain across the one-vs-rest sub-models", {
  set.seed(1)
  x <- as.matrix(iris[, 1:4])
  fit <- fastgbm(x, y = iris$Species, ntrees = 20, verbose = FALSE)
  imp <- importance(fit)
  expect_setequal(imp$feature, colnames(x))
  expect_true(all(imp$gain >= 0))
  expect_true(is.unsorted(-imp$gain) == FALSE)
})

test_that("print.fastgbm_multiclass runs without error", {
  set.seed(1)
  x <- as.matrix(iris[, 1:4])
  fit <- fastgbm(x, y = iris$Species, ntrees = 10, verbose = FALSE)
  expect_output(print(fit), "fastgbm multiclass model")
})

test_that("the formula interface works for multiclass", {
  set.seed(1)
  fit <- fastgbm(Species ~ ., data = iris, ntrees = 20, verbose = FALSE)
  expect_s3_class(fit, "fastgbm_multiclass")
  pred <- predict(fit, iris, type = "class")
  expect_gt(mean(as.character(pred) == as.character(iris$Species)), 0.8)
})

test_that("validation/early_stopping are honored per one-vs-rest sub-model", {
  set.seed(1)
  n <- 200
  x <- matrix(rnorm(n * 4), n, 4)
  cls <- sample(c("a", "b", "c"), n, replace = TRUE)
  tr <- 1:150
  va <- 151:200
  fit <- fastgbm(
    x[tr, ], y = factor(cls[tr]), ntrees = 300, verbose = FALSE,
    validation = list(x = x[va, ], y = factor(cls[va], levels = levels(factor(cls)))),
    early_stopping = 5
  )
  expect_true(all(fit$ntrees <= 300))
})

test_that("multiclass requires y with 3+ levels to auto-trigger; 2-level factors stay binary", {
  set.seed(1)
  x <- matrix(rnorm(40 * 3), 40, 3)
  y2 <- factor(sample(c("x", "y"), 40, replace = TRUE))
  fit <- fastgbm(x, y = y2, ntrees = 10, verbose = FALSE)
  expect_equal(fit$objective, "binary")
  expect_s3_class(fit, "fastgbm")
  expect_false(inherits(fit, "fastgbm_multiclass"))
})
