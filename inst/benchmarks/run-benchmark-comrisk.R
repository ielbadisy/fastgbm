## Reproducible competing-risks benchmark: fastgbm's three `comrisk` options
## ("sh" Fine-Gray, "csh" cause-specific, "ccsh" coupled cause-specific) vs
## CoxBoost (cmprsk = "sh"), linear Fine-Gray (riskRegression::FGR), linear
## cause-specific Cox (riskRegression::CSC), and a competing-risks random
## survival forest (randomForestSRC::rfsrc).
##
## Two causes with exponential cause-specific hazards, x1..x10 ~ N(0, 1)
## iid (x6..x10 pure noise), independent exponential censoring:
##   linear:    lp1 = 0.8 x1 - 0.5 x2 + 0.5 x3,  lp2 = -0.6 x1 + 0.7 x4
##   nonlinear: lp1 = 1.2 sin(2 x1) + 0.9 1(x2 > 0) x3 + 0.4 x3^2 - 0.4,
##              lp2 = 0.9 |x4| - 0.7 x1 x5 - 0.6
## n_train in {1000, 4000}, independent test set of 4000, 10 replicates per
## condition (mcstatsim::runsim()). Cause-1 cumulative incidence is scored at
## the 25/50/75% quantiles of the training event times with IPCW AUC and
## Brier score (riskRegression::Score()); "rel_brier_gain" is the Brier
## reduction relative to the Aalen-Johansen null model. Every method runs
## single-threaded; fit_sec includes prediction on the test set. fastgbm
## uses ntrees = 1000, learning_rate = 0.03, max_depth = 3 with early
## stopping (patience 30) on a 20% split of the training data; the other
## methods use their defaults (CoxBoost stepno = 100, rfsrc ntree = 500).
##
## Run from the package root:
##   Rscript inst/benchmarks/run-benchmark-comrisk.R

suppressMessages({library(mcstatsim); library(fastgbm); library(CoxBoost)
  library(riskRegression); library(prodlim); library(survival); library(randomForestSRC)})

gen <- function(n, dgp, p = 10) {
  x <- matrix(rnorm(n * p), n, p, dimnames = list(NULL, paste0("x", 1:p)))
  if (dgp == "linear") {
    lp1 <- 0.8 * x[, 1] - 0.5 * x[, 2] + 0.5 * x[, 3]
    lp2 <- -0.6 * x[, 1] + 0.7 * x[, 4]
  } else {
    lp1 <- 1.2 * sin(2 * x[, 1]) + 0.9 * (x[, 2] > 0) * x[, 3] + 0.4 * x[, 3]^2 - 0.4
    lp2 <- 0.9 * abs(x[, 4]) - 0.7 * x[, 1] * x[, 5] - 0.6
  }
  t1 <- rexp(n, 0.1 * exp(lp1)); t2 <- rexp(n, 0.08 * exp(lp2)); cc <- rexp(n, 0.04)
  time <- pmin(t1, t2, cc)
  status <- ifelse(cc < pmin(t1, t2), 0L, ifelse(t1 < t2, 1L, 2L))
  data.frame(time = time, status = status, x)
}

sim_func <- function(n_train, dgp) {
  options(rf.cores = 1, mc.cores = 1)
  tr <- gen(n_train, dgp); te <- gen(4000, dgp)
  xtr <- as.matrix(tr[, -(1:2)]); xte <- as.matrix(te[, -(1:2)])
  tt <- unname(quantile(tr$time[tr$status > 0], c(0.25, 0.5, 0.75)))
  iv <- sample(n_train, round(0.2 * n_train)); it <- setdiff(seq_len(n_train), iv)
  fml <- as.formula(paste("Hist(time, status) ~", paste(colnames(xtr), collapse = " + ")))
  preds <- list(); secs <- c()
  timed <- function(name, expr) {
    st <- system.time(p <- expr)[["elapsed"]]
    preds[[name]] <<- p; secs[[name]] <<- st
  }
  for (m in c("sh", "csh", "ccsh")) {
    timed(paste0("fastgbm_", m), {
      f <- fastgbm(xtr[it, ], time = tr$time[it], status = tr$status[it], comrisk = m,
                   ntrees = 1000, learning_rate = 0.03, max_depth = 3, threads = 1L,
                   validation = list(x = xtr[iv, ], time = tr$time[iv], status = tr$status[iv]),
                   early_stopping = 30, verbose = FALSE)
      predict(f, xte, times = tt)[[1]]
    })
  }
  timed("CoxBoost_sh", {
    f <- CoxBoost(tr$time, tr$status, xtr, stepno = 100, cmprsk = "sh")
    predict(f, newdata = xte, type = "CIF", times = tt)
  })
  timed("FineGray_linear", {
    f <- FGR(fml, data = tr, cause = 1)
    predictRisk(f, newdata = te, times = tt, cause = 1)
  })
  timed("CSC_linear", {
    f <- CSC(fml, data = tr, cause = 1)
    predictRisk(f, newdata = te, times = tt, cause = 1)
  })
  timed("rfsrc", {
    f <- rfsrc(Surv(time, status) ~ ., data = tr, ntree = 500)
    predictRisk(f, newdata = te, times = tt, cause = 1)
  })
  sc <- Score(lapply(preds, as.matrix), formula = Hist(time, status) ~ 1, data = te, times = tt,
              cause = 1, metrics = c("auc", "brier"), null.model = TRUE, se.fit = FALSE)
  auc <- aggregate(AUC ~ model, sc$AUC$score, mean)
  br <- aggregate(Brier ~ model, sc$Brier$score, mean)
  null_brier <- br$Brier[br$model == "Null model"]
  out <- merge(auc, br)
  out$model <- as.character(out$model)
  out$rel_brier_gain <- 1 - out$Brier / null_brier
  out$fit_sec <- unlist(secs[out$model])
  out
}

grid <- expand.grid(n_train = c(1000, 4000), dgp = c("linear", "nonlinear"), stringsAsFactors = FALSE)
res <- runsim(10, grid, sim_func, num_cores = 8, seed = 2026, on_error = "warn", show_progress = FALSE)
write.csv(res, "inst/benchmarks/comrisk-benchmark-results.csv", row.names = FALSE)

summ <- aggregate(cbind(AUC, Brier, rel_brier_gain, fit_sec) ~ dgp + n_train + model, res, mean)
summ <- summ[order(summ$dgp, summ$n_train, -summ$AUC), ]
write.csv(summ, "inst/benchmarks/comrisk-benchmark-summary.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "inst/benchmarks/comrisk-session-info.txt")
