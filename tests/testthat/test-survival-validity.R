## Scientific-validity checks for the survival objectives, comparing against
## `survival::coxph`/`basehaz` on synthetic proportional-hazards data.

test_that("Breslow baseline cumulative hazard matches survival::basehaz exactly given the same linear predictor", {
  skip_if_not_installed("survival")
  set.seed(5)
  n <- 50
  x1 <- rnorm(n)
  time <- rexp(n, rate = exp(0.7 * x1) * 0.1)
  cens <- rexp(n, rate = 0.05)
  status <- as.integer(time <= cens)
  time <- pmin(time, cens)
  dat <- data.frame(time = time, status = status, x1 = x1)

  cfit <- survival::coxph(survival::Surv(time, status) ~ x1, data = dat, ties = "breslow")
  lp_uncentered <- as.numeric(x1 * coef(cfit))
  bh <- survival::basehaz(cfit, centered = FALSE)

  myb <- fastgbm:::fastgbm_survival_baseline(time, status, lp_uncentered)

  common <- intersect(round(bh$time, 6), round(myb$times, 6))
  expect_equal(length(common), length(myb$times))
  idx1 <- match(common, round(bh$time, 6))
  idx2 <- match(common, round(myb$times, 6))
  expect_equal(myb$cumhaz[idx2], bh$hazard[idx1], tolerance = 1e-8)
})

test_that("baseline cumulative hazard is non-decreasing and risk set shrinks over time", {
  set.seed(6)
  n <- 40
  time <- sort(rexp(n))
  status <- rbinom(n, 1, 0.7)
  lp <- rnorm(n, sd = 0.3)
  b <- fastgbm:::fastgbm_survival_baseline(time, status, lp)
  expect_true(all(diff(b$cumhaz) >= 0))
  # a later event time must never have a larger at-risk set than an earlier one
  et <- exp(lp)
  risk_at <- function(t) sum(et[time >= t])
  risks <- vapply(b$times, risk_at, numeric(1))
  expect_true(all(diff(risks) <= 1e-8))
})

test_that("fastgbm Cox risk score is directionally concordant with coxph on a proportional-hazards DGP", {
  skip_if_not_installed("survival")
  set.seed(7)
  n <- 300
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  lp_true <- 0.9 * x1 - 0.6 * x2
  time <- rexp(n, rate = exp(lp_true) * 0.2)
  cens <- rexp(n, rate = 0.1)
  status <- as.integer(time <= cens)
  time <- pmin(time, cens)
  x <- cbind(x1 = x1, x2 = x2)

  cfit <- survival::coxph(survival::Surv(time, status) ~ x1 + x2)
  cox_lp <- predict(cfit, type = "lp")

  fit <- fastgbm(x, time = time, status = status, objective = "cox",
                 ntrees = 100L, learning_rate = 0.05, max_depth = 3L, min_node_size = 10L,
                 seed = 1L, verbose = FALSE)
  fastgbm_lp <- predict(fit, x, type = "link")

  # A well-specified boosted Cox model should rank observations similarly to the
  # (correctly-specified, linear) coxph model, and both should discriminate well.
  expect_gt(cor(fastgbm_lp, cox_lp, method = "spearman"), 0.6)
  expect_gt(fastgbm:::survival_cindex(time, status, fastgbm_lp), 0.65)
  expect_gt(fastgbm:::survival_cindex(time, status, cox_lp), 0.65)
})
