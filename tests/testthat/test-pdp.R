test_that("pdp() recovers a monotonic signal on a strongly monotonic feature", {
  set.seed(1)
  n <- 300
  x <- matrix(rnorm(n * 2), ncol = 2, dimnames = list(NULL, c("x1", "x2")))
  lp_true <- 2 * x[, 1] - 1 * x[, 2]
  time <- rexp(n, rate = exp(lp_true) * 0.2)
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 150L,
                learning_rate = 0.1, max_depth = 3L, verbose = FALSE)
  p <- pdp(fit, "x1", data = as.data.frame(x), grid_resolution = 15L)
  expect_s3_class(p, "fastgbm_pdp")
  # Exponential-time Cox data is noisy at n=300, so a handful of grid steps can
  # locally dip; require the overall trend to be strongly monotonic instead of
  # every single step.
  expect_gt(cor(p$x, p$yhat, method = "spearman"), 0.9)
  expect_gt(cor(p$x, p$yhat), 0.9)
})

test_that("pdp() matches the `pdp` package's partial() for a matrix-fitted model", {
  skip_if_not_installed("pdp")
  set.seed(1)
  n <- 200
  x <- matrix(rnorm(n * 2), ncol = 2, dimnames = list(NULL, c("x1", "x2")))
  lp_true <- 2 * x[, 1] - 1 * x[, 2]
  time <- rexp(n, rate = exp(lp_true) * 0.2)
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 100L,
                learning_rate = 0.1, max_depth = 3L, verbose = FALSE)
  mine <- pdp(fit, "x1", data = as.data.frame(x), grid_resolution = 10L)

  # `pdp::partial()` defaults to an evenly-spaced grid; request its quantile-based
  # grid (matching pdp()'s own grid construction) so the two are comparable, and
  # explicitly request the link scale to match pdp()'s default.
  pf <- function(object, newdata) predict(object, as.matrix(newdata), type = "link")
  ref <- pdp::partial(fit, pred.var = "x1", train = as.data.frame(x), pred.fun = pf,
                      quantiles = TRUE, probs = seq(0, 1, length.out = 10L))
  ref_agg <- aggregate(yhat ~ x1, data = ref, FUN = mean)

  m <- merge(mine, ref_agg, by.x = "x", by.y = "x1")
  expect_equal(nrow(m), nrow(mine))
  expect_equal(m$yhat.x, m$yhat.y, tolerance = 1e-8)
})

test_that("pdp() works on a formula-fitted model with a factor predictor", {
  skip_if_not_installed("survival")
  set.seed(2)
  n <- 250
  dat <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = factor(sample(c("a", "b"), n, TRUE)))
  lp_true <- 2 * dat$x1 + ifelse(dat$x3 == "b", 1, 0)
  dat$time <- rexp(n, rate = exp(lp_true) * 0.2)
  dat$status <- 1L
  fit <- fastgbm(survival::Surv(time, status) ~ x1 + x2 + x3, data = dat, objective = "cox",
                ntrees = 100L, learning_rate = 0.1, max_depth = 3L, verbose = FALSE)
  p <- pdp(fit, "x1", data = dat, grid_resolution = 8L)
  expect_true(all(is.finite(p$yhat)))
  expect_gt(cor(p$x, p$yhat), 0.9)
})

test_that("pdp() defaults to the linear predictor", {
  skip_if_not_installed("survival")
  dat <- na.omit(as.data.frame(survival::lung[, c("time", "status", "age", "sex", "ph.ecog")]))
  x <- model.matrix(~ age + sex + ph.ecog - 1, dat)
  fit <- fastgbm(x, time = dat$time, status = dat$status, objective = "cox",
                ntrees = 20L, learning_rate = 0.1, max_depth = 2L, min_node_size = 10L,
                seed = 1L, verbose = FALSE)
  p <- pdp(fit, "age", data = as.data.frame(x), grid_resolution = 6L)
  expect_equal(attr(p, "pdp_type"), "link")
  expect_true(all(is.finite(p$yhat)))
})

test_that("plot.fastgbm_pdp() runs without error", {
  set.seed(3)
  n <- 100
  x <- matrix(rnorm(n), ncol = 1, dimnames = list(NULL, "x1"))
  time <- exp(x[, 1] + rnorm(n, sd = 0.2))
  status <- rep(1L, n)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 20L, verbose = FALSE)
  p <- pdp(fit, "x1", data = as.data.frame(x), grid_resolution = 5L)
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_silent(invisible(plot(p)))
})

test_that("pdp() accepts a matrix and matches the data frame result", {
  set.seed(4)
  x <- matrix(rnorm(300), ncol = 3, dimnames = list(NULL, c("x1", "x2", "x3")))
  fit <- fastgbm(x, time = rexp(100, exp(x[, 1])), status = rep(1L, 100), ntrees = 30L, verbose = FALSE)
  pm <- pdp(fit, "x2", x, grid_resolution = 6L)
  expect_s3_class(pm, "fastgbm_pdp")
  expect_equal(unclass(pm)$yhat, pdp(fit, "x2", as.data.frame(x), grid_resolution = 6L)$yhat)
  expect_error(pdp(fit, "nope", x), "not found")
})

test_that("plot.fastgbm_pdp() returns an unprinted ggplot with editable labels", {
  skip_if_not_installed("ggplot2")
  set.seed(5)
  x <- matrix(rnorm(200), ncol = 2, dimnames = list(NULL, c("age", "x2")))
  fit <- fastgbm(x, time = rexp(100, exp(x[, 1])), status = rep(1L, 100), ntrees = 20L, verbose = FALSE)
  p <- pdp(fit, "age", x, grid_resolution = 5L)
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  n0 <- grDevices::dev.cur()
  g <- plot(p)
  expect_s3_class(g, "ggplot")
  expect_identical(g$labels$x, "age")
  expect_null(g$labels$title)
  g2 <- plot(p, xlab = "Age (years)", title = "PD", caption = "c") + ggplot2::labs(y = "log hazard")
  expect_identical(g2$labels$x, "Age (years)")
  expect_identical(g2$labels$y, "log hazard")
  expect_identical(g2$labels$title, "PD")
  expect_null(plot(p, title = NULL)$labels$title)
})

test_that("pdp() on the survival scale averages predicted survival at one horizon", {
  set.seed(3)
  x <- matrix(rnorm(400), ncol = 2, dimnames = list(NULL, c("x1", "x2")))
  fit <- fastgbm(x, time = rexp(200, exp(x[, 1])), status = rep(1L, 200), ntrees = 30L, verbose = FALSE)
  p <- pdp(fit, "x1", x, grid_resolution = 5L, type = "survival", times = 0.5)
  own <- vapply(p$x, function(g) {
    d <- x; d[, "x1"] <- g; mean(predict(fit, d, type = "survival", times = 0.5))
  }, 0)
  expect_equal(p$yhat, own)
  expect_lt(cor(p$x, p$yhat), 0)
  expect_identical(attr(p, "pdp_type"), "survival at t = 0.5")
})
