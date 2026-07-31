## Verify the compiled gradient/Hessian kernels (exposed via the internal
## `fastgbm_grad_hess()` test hook, src/fastgbm.cpp) against finite-difference
## derivatives of the *stated* loss for each objective. This checks the actual
## compiled math, not a hand-written R re-implementation of it.

numgrad <- function(loss_fun, pred, i, eps = 1e-6) {
  p1 <- pred; p1[i] <- p1[i] + eps
  p0 <- pred; p0[i] <- p0[i] - eps
  (loss_fun(p1) - loss_fun(p0)) / (2 * eps)
}

numhess <- function(loss_fun, pred, i, eps = 1e-4) {
  p1 <- pred; p1[i] <- p1[i] + eps
  p0 <- pred; p0[i] <- p0[i] - eps
  pm <- pred
  (loss_fun(p1) - 2 * loss_fun(pm) + loss_fun(p0)) / (eps^2)
}

test_that("Cox (Breslow) gradient/Hessian match finite differences of the negative partial log-likelihood", {
  set.seed(2)
  n <- 12
  time <- c(2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 8, 9)
  status <- c(1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1)
  pred <- rnorm(n, sd = 0.5)

  neg_log_partial_lik <- function(p) {
    -sum(vapply(seq_len(n), function(i) {
      if (status[i] != 1) return(0)
      risk_set <- which(time >= time[i])
      p[i] - log(sum(exp(p[risk_set])))
    }, numeric(1)))
  }

  gh <- fastgbm:::fastgbm_grad_hess(pred, time = time, status = status, objective = "cox")
  for (i in seq_along(pred)) {
    expect_equal(gh$grad[i], numgrad(neg_log_partial_lik, pred, i), tolerance = 1e-4)
    expect_equal(gh$hess[i], numhess(neg_log_partial_lik, pred, i), tolerance = 5e-3)
  }
})

test_that("AFT (normal) gradient/Hessian match finite differences of the negative log-likelihood", {
  set.seed(3)
  n <- 10
  time <- exp(rnorm(n, sd = 0.3))
  status <- c(1, 1, 0, 1, 0, 1, 1, 0, 1, 1)
  pred <- rnorm(n, sd = 0.4)
  sigma <- 1

  neg_log_lik <- function(p) {
    logt <- log(time)
    z <- (logt - p) / sigma
    ll <- ifelse(
      status == 1,
      dnorm(z, log = TRUE) - log(sigma * time),
      pnorm(z, lower.tail = FALSE, log.p = TRUE)
    )
    -sum(ll)
  }

  gh <- fastgbm:::fastgbm_grad_hess(pred, time = time, status = status, objective = "aft")
  for (i in seq_along(pred)) {
    expect_equal(gh$grad[i], numgrad(neg_log_lik, pred, i), tolerance = 1e-4)
    expect_equal(gh$hess[i], numhess(neg_log_lik, pred, i), tolerance = 5e-3)
  }
})

test_that("piecewise-exponential (Poisson-trick) gradient/Hessian match finite differences", {
  ## `time`/`status` here play the role of exposure/event on person-time-
  ## expanded rows (see R/pexp.R); the loss is an ordinary Poisson deviance
  ## with the exposure as an offset.
  set.seed(4)
  n <- 15
  exposure <- runif(n, 0.2, 3)
  event <- rbinom(n, 1, 0.4)
  pred <- rnorm(n, sd = 0.4)

  neg_poisson_loglik <- function(p) {
    mean_i <- exposure * exp(p)
    sum(mean_i - event * p)  # dropping the event*log(exposure) constant, as in compute_pexp_loss
  }

  gh <- fastgbm:::fastgbm_grad_hess(pred, time = exposure, status = event, objective = "pexp")
  for (i in seq_along(pred)) {
    expect_equal(gh$grad[i], numgrad(neg_poisson_loglik, pred, i), tolerance = 1e-4)
    expect_equal(gh$hess[i], numhess(neg_poisson_loglik, pred, i), tolerance = 5e-3)
  }
})

test_that("regression (squared error) gradient/Hessian match finite differences", {
  ## `y` (the regression target) travels in the "time" argument slot, same
  ## convention as pexp's exposure; "status" is unused for this objective.
  set.seed(5)
  n <- 10
  y <- rnorm(n)
  pred <- rnorm(n, sd = 0.4)

  half_sq_err <- function(p) sum(0.5 * (p - y)^2)

  gh <- fastgbm:::fastgbm_grad_hess(pred, time = y, status = integer(n), objective = "regression")
  for (i in seq_along(pred)) {
    expect_equal(gh$grad[i], numgrad(half_sq_err, pred, i), tolerance = 1e-4)
    expect_equal(gh$hess[i], numhess(half_sq_err, pred, i), tolerance = 5e-3)
  }
})

test_that("binary (logistic) gradient/Hessian match finite differences", {
  set.seed(6)
  n <- 10
  y <- rbinom(n, 1, 0.5)
  pred <- rnorm(n, sd = 0.4)

  neg_log_lik <- function(p) {
    prob <- 1 / (1 + exp(-p))
    -sum(y * log(prob) + (1 - y) * log(1 - prob))
  }

  gh <- fastgbm:::fastgbm_grad_hess(pred, time = y, status = integer(n), objective = "binary")
  for (i in seq_along(pred)) {
    expect_equal(gh$grad[i], numgrad(neg_log_lik, pred, i), tolerance = 1e-4)
    expect_equal(gh$hess[i], numhess(neg_log_lik, pred, i), tolerance = 5e-3)
  }
})
