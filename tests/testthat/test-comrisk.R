## Competing risks (`comrisk`): the Fine-Gray objective and the cumulative
## incidence predictions are checked against CoxBoost, whose three options
## ("sh", "csh", "ccsh") fastgbm follows.

numgrad <- function(loss_fun, pred, i, eps = 1e-6) {
  p1 <- pred; p1[i] <- p1[i] + eps
  p0 <- pred; p0[i] <- p0[i] - eps
  (loss_fun(p1) - loss_fun(p0)) / (2 * eps)
}

numhess <- function(loss_fun, pred, i, eps = 1e-4) {
  p1 <- pred; p1[i] <- p1[i] + eps
  p0 <- pred; p0[i] <- p0[i] - eps
  (loss_fun(p1) - 2 * loss_fun(pred) + loss_fun(p0)) / (eps^2)
}

comrisk_data <- function(n = 300, seed = 1) {
  set.seed(seed)
  x <- matrix(rnorm(n * 4), n, 4, dimnames = list(NULL, paste0("x", 1:4)))
  t1 <- rexp(n, 0.1 * exp(0.8 * x[, 1]))
  t2 <- rexp(n, 0.1 * exp(-0.6 * x[, 2]))
  cens <- rexp(n, 0.05)
  time <- pmin(t1, t2, cens)
  status <- ifelse(cens < pmin(t1, t2), 0L, ifelse(t1 < t2, 1L, 2L))
  list(x = x, time = time, status = status)
}

test_that("Fine-Gray gradient matches finite differences of CoxBoost's IPCW partial likelihood", {
  skip_if_not_installed("CoxBoost")
  set.seed(5)
  n <- 40
  time <- round(rexp(n), 6)
  status <- sample(0:2, n, TRUE, prob = c(0.3, 0.4, 0.3))
  pred <- rnorm(n, sd = 0.5)
  W <- CoxBoost:::efron.weightmat(time, status, 1)
  ev <- which(status == 1)
  loss <- function(p) -sum(p[ev] - log(colSums(W * exp(p))))

  ipcw <- fastgbm:::fastgbm_eval_step(fastgbm:::fastgbm_censoring_km(time, status), time)
  gh <- fastgbm:::fastgbm_grad_hess(pred, time, status, "finegray", ipcw)
  for (i in seq_len(n)) {
    expect_equal(gh$grad[i], numgrad(loss, pred, i), tolerance = 1e-4)
    expect_equal(gh$hess[i], numhess(loss, pred, i), tolerance = 5e-3)
  }

  o <- order(time[ev])
  lambda_cb <- cumsum((1 / colSums(W * exp(pred)))[o])
  b <- fastgbm:::fastgbm_finegray_baseline(time, status, ipcw, pred)
  expect_equal(b$cumhaz, lambda_cb, tolerance = 1e-10)
})

test_that("Fine-Gray reduces to Cox when there are no competing events", {
  set.seed(6)
  time <- c(2, 2, 3, 4, 4, 5, 6, 7, 8, 9)
  status <- c(1, 0, 1, 1, 0, 0, 1, 1, 0, 1)
  pred <- rnorm(10, sd = 0.5)
  ipcw <- fastgbm:::fastgbm_eval_step(fastgbm:::fastgbm_censoring_km(time, status), time)
  fg <- fastgbm:::fastgbm_grad_hess(pred, time, status, "finegray", ipcw)
  cox <- fastgbm:::fastgbm_grad_hess(pred, time, status, "cox")
  expect_equal(fg$grad, cox$grad, tolerance = 1e-12)
  expect_equal(fg$hess, cox$hess, tolerance = 1e-12)
})

test_that("cumulative incidence matches CoxBoost's predict(type = \"CIF\")", {
  skip_if_not_installed("CoxBoost")
  set.seed(5)
  n <- 80
  x <- matrix(rnorm(n * 2), n)
  time <- round(rexp(n), 6)
  status <- sample(0:2, n, TRUE, prob = c(0.3, 0.4, 0.3))
  tt <- c(0.1, 0.5, 1, 2)

  cb <- CoxBoost::CoxBoost(time, status, x, stepno = 0, cmprsk = "csh")
  cif_cb <- predict(cb, newdata = x[1:5, ], type = "CIF", times = tt)
  bl <- lapply(1:2, function(k) fastgbm:::fastgbm_survival_baseline(time, as.integer(status == k), rep(0, n)))
  cif <- fastgbm:::fastgbm_cif_cause_specific(matrix(0, 5, 2), bl, tt)
  expect_equal(unname(cif[[1]]), unname(cif_cb[[1]]), tolerance = 1e-10)
  expect_equal(unname(cif[[2]]), unname(cif_cb[[2]]), tolerance = 1e-10)

  cb <- CoxBoost::CoxBoost(time, status, x, stepno = 0, cmprsk = "sh")
  cif_cb <- predict(cb, newdata = x[1:5, ], type = "CIF", times = tt)
  ipcw <- fastgbm:::fastgbm_eval_step(fastgbm:::fastgbm_censoring_km(time, status), time)
  b <- fastgbm:::fastgbm_finegray_baseline(time, status, ipcw, rep(0, n))
  H <- fastgbm:::fastgbm_eval_step(list(times = b$times, surv = b$cumhaz), tt, at_zero = 0)
  expect_equal(1 - exp(-H), unname(cif_cb[1, ]), tolerance = 1e-10)
})

test_that("all three comrisk options fit and give valid cumulative incidences", {
  d <- comrisk_data()
  tt <- c(1, 3, 6, 12)
  for (m in c("sh", "csh", "ccsh")) {
    fit <- fastgbm(d$x, time = d$time, status = d$status, comrisk = m,
                   ntrees = 40, learning_rate = 0.05, max_depth = 2, verbose = FALSE)
    expect_s3_class(fit, "fastgbm_comrisk")
    expect_equal(fit$comrisk, m)
    cif <- predict(fit, d$x[1:20, ], times = tt)
    expect_length(cif, if (m == "sh") 1L else 2L)
    for (ci in cif) {
      expect_equal(dim(ci), c(20L, length(tt)))
      expect_true(all(ci >= 0 & ci <= 1))
      expect_true(all(apply(ci, 1, diff) >= -1e-12))
    }
    if (m != "sh") {
      expect_true(all(cif[[1]] + cif[[2]] <= 1 + 1e-12))
    }
    lp <- predict(fit, d$x, type = "link")
    expect_equal(dim(lp), c(nrow(d$x), length(cif)))
    expect_gt(cor(lp[, 1], d$x[, 1]), 0.5)
    expect_s3_class(importance(fit), "data.frame")
  }
})

test_that("comrisk = \"csh\" equals separate cause-specific Cox fits", {
  d <- comrisk_data()
  fit <- fastgbm(d$x, time = d$time, status = d$status, comrisk = "csh",
                 ntrees = 20, max_depth = 2, seed = 3L, verbose = FALSE)
  for (k in 1:2) {
    cox <- fastgbm(d$x, time = d$time, status = as.integer(d$status == k), objective = "cox",
                   ntrees = 20, max_depth = 2, seed = 3L + k - 1L, verbose = FALSE)
    expect_equal(unname(predict(fit, d$x, type = "link")[, k]),
                 unname(predict(cox, d$x, type = "link")))
  }
})

test_that("comrisk = \"ccsh\" grows one shared tree per round with per-cause leaves", {
  d <- comrisk_data()
  fit <- fastgbm(d$x, time = d$time, status = d$status, comrisk = "ccsh",
                 ntrees = 10, max_depth = 3, verbose = FALSE)
  t1 <- fit$fits[[1]]$trees
  t2 <- fit$fits[[2]]$trees
  for (b in seq_along(t1)) {
    expect_identical(t1[[b]]$feature, t2[[b]]$feature)
    expect_identical(t1[[b]]$threshold, t2[[b]]$threshold)
  }
  expect_false(isTRUE(all.equal(t1[[1]]$value, t2[[1]]$value)))
})

test_that("comrisk validates its input and supports Surv and early stopping", {
  skip_if_not_installed("survival")
  d <- comrisk_data()
  expect_error(fastgbm(d$x, time = d$time, status = d$status, objective = "aft", verbose = FALSE),
               "only available")
  expect_error(fastgbm(d$x, time = d$time, status = d$status, cause = 3, verbose = FALSE),
               "`cause`")
  expect_error(fastgbm(d$x, time = d$time, status = as.integer(d$status > 0), comrisk = "sh", verbose = FALSE),
               "at least two causes")

  df <- data.frame(time = d$time, ev = factor(d$status, 0:2, c("censor", "relapse", "death")), d$x)
  fit <- fastgbm(survival::Surv(time, ev) ~ ., data = df, cause = "death", ntrees = 10, verbose = FALSE)
  expect_equal(fit$comrisk, "sh")
  expect_equal(fit$all_causes, c("relapse", "death"))
  expect_named(predict(fit, df[1:3, ], times = 5), "death")

  idx <- 1:200
  for (m in c("sh", "csh", "ccsh")) {
    fit <- fastgbm(d$x[idx, ], time = d$time[idx], status = d$status[idx], comrisk = m,
                   ntrees = 300, learning_rate = 0.2,
                   validation = list(x = d$x[-idx, ], time = d$time[-idx], status = d$status[-idx]),
                   early_stopping = 5, verbose = FALSE)
    expect_true(all(fit$ntrees < 300))
  }
})
