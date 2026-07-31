## Piecewise-exponential (PEM) objective: person-time expansion + the joint
## hazard-over-time boosting architecture (R/pexp.R). See test-gradients.R for
## the finite-difference gradient/Hessian check; these tests cover the
## higher-level expansion/prediction machinery built on top of it.

test_that("person-time expansion preserves event counts and produces positive exposures", {
  set.seed(1)
  n <- 40
  x <- matrix(rnorm(n * 2), ncol = 2)
  time <- rexp(n, rate = 0.3)
  status <- rbinom(n, 1, 0.7)
  cuts <- fastgbm:::fastgbm_pexp_cutpoints(time, status, bins = 6L)

  expanded <- fastgbm:::fastgbm_expand_person_time(x, time, status, cuts)
  expect_equal(sum(expanded$event), sum(status))
  expect_true(all(expanded$exposure > 0))
  expect_equal(ncol(expanded$x), ncol(x) + 1L)
  expect_equal(nrow(expanded$x), length(expanded$exposure))
  expect_equal(nrow(expanded$x), length(expanded$interval))
  expect_true(all(expanded$interval >= 1L & expanded$interval <= 6L))
  # every subject appears at least once
  expect_equal(length(unique(expanded$subject)), n)
})

test_that("a Poisson GLM on the expanded data recovers a covariate effect close to coxph's", {
  skip_if_not_installed("survival")
  set.seed(2)
  n <- 400
  x1 <- rnorm(n)
  lp_true <- 0.7 * x1
  time <- rexp(n, rate = exp(lp_true) * 0.2)
  status <- rbinom(n, 1, 0.85)
  cuts <- fastgbm:::fastgbm_pexp_cutpoints(time, status, bins = 10L)
  expanded <- fastgbm:::fastgbm_expand_person_time(matrix(x1, ncol = 1), time, status, cuts)

  df <- data.frame(x1 = expanded$x[, 1], interval = factor(expanded$interval),
                   exposure = expanded$exposure, event = expanded$event)
  glm_fit <- glm(event ~ x1 + interval, offset = log(exposure), family = poisson(), data = df)
  cfit <- survival::coxph(survival::Surv(time, status) ~ x1)

  expect_equal(unname(coef(glm_fit)["x1"]), unname(coef(cfit)["x1"]), tolerance = 0.15)
})

test_that("fastgbm(objective = \"pexp\") fits, predicts finitely, and discriminates reasonably", {
  skip_if_not_installed("survival")
  set.seed(3)
  n <- 300
  x <- matrix(rnorm(n * 3), ncol = 3, dimnames = list(NULL, c("x1", "x2", "x3")))
  lp <- 0.7 * x[, 1] - 0.4 * x[, 2]
  time <- rexp(n, rate = exp(lp) * 0.2)
  status <- rbinom(n, 1, 0.85)

  fit <- fastgbm(x, time = time, status = status, objective = "pexp", ntrees = 100L,
                learning_rate = 0.1, max_depth = 3L, min_node_size = 10L, seed = 1L, verbose = FALSE)

  link <- predict(fit, x, type = "link")
  resp <- predict(fit, x, type = "response")
  expect_true(all(is.finite(link)))
  expect_true(all(resp > 0))
  expect_equal(resp, exp(link), tolerance = 1e-8)

  m <- metrics(fit, y = survival::Surv(time, status))
  expect_equal(m$metric, "cindex")
  expect_gt(m$value, 0.65)

  cfit <- survival::coxph(survival::Surv(time, status) ~ x1 + x2 + x3, data = as.data.frame(x))
  expect_gt(cor(link, predict(cfit, type = "lp"), method = "spearman"), 0.7)
})

test_that("pexp survival curves are monotonic, bounded, and match exp(-cumhaz)", {
  set.seed(4)
  n <- 150
  x <- matrix(rnorm(n * 2), ncol = 2)
  time <- rexp(n, rate = 0.2)
  status <- rbinom(n, 1, 0.8)
  fit <- fastgbm(x, time = time, status = status, objective = "pexp", ntrees = 50L,
                max_depth = 3L, min_node_size = 10L, seed = 1L, verbose = FALSE)

  times <- c(1, 2, 5, 10, 15)
  s <- predict(fit, x[1:10, ], type = "survival", times = times)
  expect_equal(dim(s), c(10L, length(times)))
  expect_true(all(s >= 0 & s <= 1))
  expect_true(all(apply(s, 1, function(r) all(diff(r) <= 1e-8))))

  H <- fastgbm:::fastgbm_pexp_cumhaz(fit, x[1:10, ], times)
  expect_equal(s, exp(-H))
})

test_that("pexp handles missing covariates natively", {
  set.seed(5)
  n <- 100
  x <- matrix(rnorm(n * 2), ncol = 2)
  x[sample.int(n, 10), 1] <- NA
  time <- rexp(n, rate = 0.2)
  status <- rbinom(n, 1, 0.8)
  fit <- fastgbm(x, time = time, status = status, objective = "pexp", ntrees = 30L,
                max_depth = 3L, min_node_size = 10L, verbose = FALSE)
  expect_true(all(is.finite(predict(fit, x, type = "link"))))
})

test_that("pexp works with early stopping", {
  set.seed(6)
  n <- 300
  x <- matrix(rnorm(n * 2), ncol = 2)
  time <- rexp(n, rate = 0.2)
  status <- rbinom(n, 1, 0.8)
  va_x <- matrix(rnorm(100 * 2), ncol = 2)
  va_time <- rexp(100, rate = 0.2)
  va_status <- rbinom(100, 1, 0.8)

  fit <- fastgbm(x, time = time, status = status, objective = "pexp", ntrees = 300L,
                learning_rate = 0.1, max_depth = 3L, min_node_size = 10L,
                validation = list(x = va_x, time = va_time, status = va_status),
                early_stopping = 10L, threads = 1L, seed = 1L, verbose = FALSE)
  expect_lte(fit$best_iteration, fit$n_trees_grown)
  expect_true(all(is.finite(predict(fit, x, type = "link"))))
})

test_that("pexp training is deterministic across thread counts", {
  set.seed(7)
  n <- 400
  x <- matrix(rnorm(n * 6), ncol = 6)
  time <- rexp(n, rate = 0.2)
  status <- rbinom(n, 1, 0.8)

  fit1 <- fastgbm(x, time = time, status = status, objective = "pexp", ntrees = 30L,
                  max_depth = 4L, min_node_size = 5L, threads = 1L, seed = 1L, verbose = FALSE)
  fit4 <- fastgbm(x, time = time, status = status, objective = "pexp", ntrees = 30L,
                  max_depth = 4L, min_node_size = 5L, threads = 4L, seed = 1L, verbose = FALSE)
  expect_equal(predict(fit1, x, type = "link"), predict(fit4, x, type = "link"))
})

test_that("pexp serialization round trips", {
  set.seed(8)
  n <- 80
  x <- matrix(rnorm(n * 2), ncol = 2)
  time <- rexp(n, rate = 0.2)
  status <- rbinom(n, 1, 0.8)
  fit <- fastgbm(x, time = time, status = status, objective = "pexp", ntrees = 20L, verbose = FALSE)
  tmp <- tempfile(fileext = ".rds")
  save_fastgbm(fit, tmp)
  fit2 <- load_fastgbm(tmp)
  expect_equal(predict(fit, x, type = "link"), predict(fit2, x, type = "link"))
  expect_equal(predict(fit, x, type = "survival", times = c(1, 5)),
              predict(fit2, x, type = "survival", times = c(1, 5)))
})

test_that("pexp works via the formula interface", {
  skip_if_not_installed("survival")
  set.seed(9)
  n <- 150
  dat <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  dat$time <- rexp(n, rate = 0.2)
  dat$status <- rbinom(n, 1, 0.8)
  fit <- fastgbm(survival::Surv(time, status) ~ x1 + x2, data = dat, objective = "pexp",
                ntrees = 30L, verbose = FALSE)
  p <- predict(fit, dat, type = "link")
  expect_length(p, n)
  expect_true(all(is.finite(p)))
})
