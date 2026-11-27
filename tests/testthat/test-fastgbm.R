test_that("serialization round trips", {
  x <- matrix(c(1, 2, 3, 4), ncol = 1)
  time <- c(1, 2, 3, 4)
  status <- c(1, 1, 0, 1)
  fit <- fastgbm(x, time = time, status = status, objective = "cox", ntrees = 2L,
                learning_rate = 0.2, max_depth = 1L, min_node_size = 1L, seed = 3L, verbose = FALSE)
  tmp <- tempfile(fileext = ".rds")
  savefastgbm(fit, tmp)
  fit2 <- loadfastgbm(tmp)
  expect_equal(predict(fit, x), predict(fit2, x))
})

test_that("formula interface expands factors", {
  skip_if_not_installed("survival")
  dat <- data.frame(time = c(1, 2, 3, 4, 5, 6), status = c(1, 0, 1, 1, 0, 1),
                    x1 = c(1, 2, 3, 4, 5, 6), x2 = factor(c("a", "a", "b", "b", "a", "b")))
  fit <- fastgbm(survival::Surv(time, status) ~ x1 + x2, data = dat, objective = "cox",
                ntrees = 3L, learning_rate = 0.2, max_depth = 2L, min_node_size = 1L,
                seed = 4L, verbose = FALSE)
  p <- predict(fit, dat, type = "response")
  expect_length(p, nrow(dat))
})

test_that("cox survival objective trains and predicts survival curves", {
  skip_if_not_installed("survival")
  dat <- na.omit(as.data.frame(survival::lung[, c("time", "status", "age", "sex", "ph.ecog")]))
  x <- model.matrix(~ age + sex + ph.ecog - 1, dat)
  fit <- fastgbm(x, time = dat$time, status = dat$status, objective = "cox", ntrees = 5L,
                learning_rate = 0.1, max_depth = 2L, min_node_size = 5L, seed = 5L, verbose = FALSE)
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
  fit <- fastgbm(x, time = time, status = status, objective = "aft", ntrees = 5L,
                learning_rate = 0.1, max_depth = 2L, min_node_size = 5L, seed = 6L, verbose = FALSE)
  lp <- predict(fit, x, type = "link")
  s <- predict(fit, x[1:6, , drop = FALSE], type = "survival", times = c(1, 2, 3))
  expect_length(lp, nrow(x))
  expect_equal(dim(s), c(6L, 3L))
  expect_true(all(s >= 0 & s <= 1))
  expect_gt(metrics(fit, y = survival::Surv(time, status))$value, 0.4)
})

test_that("fastgbm() requires data when x is a formula", {
  skip_if_not_installed("survival")
  expect_error(fastgbm(survival::Surv(time, status) ~ x1), "`data`")
})

test_that("fastgbm() infers a regression objective for a non-Surv formula response", {
  dat <- data.frame(y = c(1, 2, 3), x1 = c(1, 2, 3))
  fit <- fastgbm(y ~ x1, data = dat, ntrees = 5, verbose = FALSE)
  expect_identical(fit$objective, "regression")
})

test_that("formula interface accepts validation data frames", {
  d <- na.omit(survival::lung); d$status <- d$status - 1; d$sex <- factor(d$sex)
  a <- d[1:120, ]; v <- d[-(1:120), ]
  f <- survival::Surv(time, status) ~ age + sex + ph.ecog
  ft <- function(val) fastgbm(f, data = a, ntrees = 30, validation = val, early_stopping = 5, verbose = FALSE)
  m1 <- ft(v)
  m2 <- ft(list(x = v[c("age", "sex", "ph.ecog")], time = v$time, status = v$status))
  x <- function(d) model.matrix(~ age + sex + ph.ecog, d)[, -1]
  m3 <- fastgbm(x(a), y = survival::Surv(a$time, a$status), ntrees = 30, verbose = FALSE,
                validation = list(x = x(v), y = survival::Surv(v$time, v$status)), early_stopping = 5)
  expect_equal(m1$validation_history, m3$validation_history)
  expect_equal(m2$validation_history, m3$validation_history)
  expect_equal(predict(m1, v, type = "link"), predict(m3, x(v), type = "link"))
})

test_that("aft_sigma sets the AFT scale in the loss and in survival predictions", {
  set.seed(4)
  x <- matrix(rnorm(400), ncol = 2, dimnames = list(NULL, c("a", "b")))
  t0 <- exp(x[, 1] + rnorm(200, 0, 0.5)); cn <- rexp(200, 0.2)
  tm <- pmin(t0, cn); st <- as.integer(t0 <= cn); v <- 151:200; a <- 1:150
  ft <- function(...) fastgbm(x[a, ], time = tm[a], status = st[a], objective = "aft", ntrees = 10L,
                              verbose = FALSE, validation = list(x = x[v, ], time = tm[v], status = st[v]),
                              early_stopping = 20L, ...)
  m1 <- ft(); m2 <- ft(aft_sigma = 0.5)
  expect_equal(m1$survival_sigma, 1); expect_equal(m2$survival_sigma, 0.5)
  expect_equal(predict(m1, x, type = "link"), predict(ft(aft_sigma = 1), x, type = "link"))
  expect_false(isTRUE(all.equal(predict(m1, x, type = "link"), predict(m2, x, type = "link"))))
  # validation loss at the best round equals the normal censored loss at sigma = 0.5
  lp <- predict(m2, x[v, ], type = "link"); z <- (log(tm[v]) - lp) / 0.5
  own <- sum(ifelse(st[v] == 1, z^2 / 2, -pnorm(z, lower.tail = FALSE, log.p = TRUE)))
  expect_equal(m2$validation_history[m2$best_iteration], own, tolerance = 1e-6)
  expect_equal(predict(m2, x[1, , drop = FALSE], type = "survival", times = 2)[1, 1],
               pnorm((log(2) - predict(m2, x[1, , drop = FALSE], type = "link")) / 0.5,
                     lower.tail = FALSE))
})

test_that("aft survival matrix has one row per subject and one column per time", {
  set.seed(5)
  x <- matrix(rnorm(400), ncol = 2, dimnames = list(NULL, c("a", "b")))
  fit <- fastgbm(x, time = exp(x[, 1] + rnorm(200)), status = rep(1L, 200), objective = "aft",
                 ntrees = 20L, verbose = FALSE)
  tt <- c(0.5, 1, 2); lp <- predict(fit, x[1:4, ], type = "link")
  expect_equal(unname(predict(fit, x[1:4, ], type = "survival", times = tt)),
               1 - pnorm(outer(-lp, log(tt), "+")))
})
