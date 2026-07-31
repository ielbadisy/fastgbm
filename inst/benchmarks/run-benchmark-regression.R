## Reproducible regression benchmark: fastgbm vs gbm, xgboost, ranger on
## Friedman's #1 data-generating process (mlbench::mlbench.friedman1()),
## the standard synthetic nonlinear-regression DGP used in the gradient
## boosting literature:
##
##   y = 10*sin(pi*x1*x2) + 20*(x3 - 0.5)^2 + 10*x4 + 5*x5 + eps,
##   x1..x10 ~ Uniform(0, 1) iid, only x1..x5 enter the mean function
##   (x6..x10 are pure noise features), eps ~ N(0, sd^2).
##
## Same "equal hyperparameter grid" regime as inst/benchmarks/run-benchmark.R
## (the survival benchmark): every package gets the same ntrees/depth/
## learning-rate/min-node-size where the parameter exists, not independently
## tuned. Training/prediction time and test-set RMSE are recorded over
## repeated random splits. A second block fits fastgbm once on a large
## sample and compares its partial-dependence shape for x1/x3/x4/x5 against
## the *true* partial dependence (computable in closed form here because the
## DGP is known), to check the ensemble recovers the right functional shapes
## (interaction-flattened for x1, quadratic for x3, linear for x4/x5) rather
## than just achieving good average error.
##
## Outputs (relative to the package root):
##   inst/benchmarks/regression-benchmark-results.csv  -- one row per (model, repeat)
##   inst/benchmarks/regression-benchmark-summary.csv  -- median/IQR + win-loss-tie per model
##   inst/benchmarks/regression-pdp.png                -- fastgbm vs true PDP, x1/x3/x4/x5
##   inst/benchmarks/regression-session-info.txt       -- reproducibility metadata
##
## Run from the package root: Rscript inst/benchmarks/run-benchmark-regression.R

suppressPackageStartupMessages({
  library(fastgbm)
  library(mlbench)
  library(gbm)
  library(xgboost)
  library(ranger)
})

SEED0 <- 20260731L
N <- 2000L
N_REPEATS <- 10L

NTREES <- 200L
MAX_DEPTH <- 5L
LR <- 0.1
MIN_NODE <- 10L

timed <- function(expr) {
  gc(FALSE)
  t <- system.time(value <- force(expr))
  list(value = value, elapsed = unname(t["elapsed"]))
}

train_test_split <- function(n, seed, p_train = 0.7) {
  set.seed(seed)
  idx <- sample.int(n)
  n_train <- max(2L, floor(p_train * n))
  list(train = idx[seq_len(n_train)], test = idx[-seq_len(n_train)])
}

rmse <- function(y, pred) sqrt(mean((y - pred)^2))

friedman1 <- function(n, seed, sd = 1) {
  set.seed(seed)
  d <- mlbench::mlbench.friedman1(n, sd = sd)
  list(x = as.matrix(d$x), y = as.numeric(d$y))
}

## ------------------------------------------------------------------ runner --

run_regression_once <- function(x, y, seed) {
  sp <- train_test_split(nrow(x), seed)
  x_tr <- x[sp$train, , drop = FALSE]; x_te <- x[sp$test, , drop = FALSE]
  y_tr <- y[sp$train]; y_te <- y[sp$test]
  colnames(x_tr) <- colnames(x_te) <- paste0("x", seq_len(ncol(x)))
  df_tr <- as.data.frame(x_tr); df_tr$.y <- y_tr
  df_te <- as.data.frame(x_te)

  rows <- list()

  # fastgbm carves its own validation split out of the training fold and uses
  # early stopping (ntrees = NTREES is a ceiling), same convention as the
  # survival benchmark; the other three packages use fixed ntrees.
  es_split <- train_test_split(nrow(x_tr), seed + 1000L, p_train = 0.85)
  x_tr2 <- x_tr[es_split$train, , drop = FALSE]; x_val <- x_tr[es_split$test, , drop = FALSE]
  y_tr2 <- y_tr[es_split$train]; y_val <- y_tr[es_split$test]

  ft <- timed(fastgbm(x_tr2, y = y_tr2, objective = "regression",
                      ntrees = NTREES, learning_rate = LR, max_depth = MAX_DEPTH,
                      min_node_size = MIN_NODE,
                      validation = list(x = x_val, y = y_val),
                      early_stopping = 20L, threads = 1L, seed = seed, verbose = FALSE))
  pt <- timed(predict(ft$value, x_te, type = "response"))
  rows$fastgbm <- data.frame(model = "fastgbm", train_sec = ft$elapsed, predict_sec = pt$elapsed,
                             metric = rmse(y_te, pt$value))

  gt <- timed(gbm::gbm(.y ~ ., data = df_tr, distribution = "gaussian",
                       n.trees = NTREES, interaction.depth = MAX_DEPTH,
                       shrinkage = LR, n.minobsinnode = MIN_NODE, bag.fraction = 1,
                       train.fraction = 1, verbose = FALSE))
  pg <- timed(predict(gt$value, newdata = df_te, n.trees = NTREES))
  rows$gbm <- data.frame(model = "gbm", train_sec = gt$elapsed, predict_sec = pg$elapsed,
                         metric = rmse(y_te, pg$value))

  dtr <- xgboost::xgb.DMatrix(x_tr, label = y_tr)
  dte <- xgboost::xgb.DMatrix(x_te)
  xt <- timed(xgboost::xgb.train(
    params = list(objective = "reg:squarederror", eta = LR, max_depth = MAX_DEPTH,
                 min_child_weight = 1, nthread = 1, verbosity = 0),
    data = dtr, nrounds = NTREES
  ))
  px <- timed(predict(xt$value, dte))
  rows$xgboost <- data.frame(model = "xgboost", train_sec = xt$elapsed, predict_sec = px$elapsed,
                             metric = rmse(y_te, px$value))

  rt <- timed(ranger::ranger(.y ~ ., data = df_tr, num.trees = NTREES,
                             max.depth = MAX_DEPTH, min.node.size = MIN_NODE,
                             num.threads = 1, seed = seed))
  pr <- timed(predict(rt$value, data = df_te))
  rows$ranger <- data.frame(model = "ranger", train_sec = rt$elapsed, predict_sec = pr$elapsed,
                            metric = rmse(y_te, pr$value$predictions))

  out <- do.call(rbind, rows)
  out$metric_name <- "rmse"
  out
}

## ------------------------------------------------------------------ driver --

all_results <- list()
message(sprintf("== friedman1 (n=%d, p=10, repeats=%d) ==", N, N_REPEATS))
for (rep in seq_len(N_REPEATS)) {
  seed <- SEED0 + rep
  d <- friedman1(N, seed = seed)
  res <- run_regression_once(d$x, d$y, seed)
  res$dataset <- "friedman1"
  res$task <- "regression"
  res$n <- N
  res$p <- ncol(d$x)
  res$rep <- rep
  all_results[[length(all_results) + 1L]] <- res
  message(sprintf("  rep %d/%d done", rep, N_REPEATS))
}
results <- do.call(rbind, all_results)
rownames(results) <- NULL

out_dir <- if (dir.exists("inst/benchmarks")) "inst/benchmarks" else "."
write.csv(results, file.path(out_dir, "regression-benchmark-results.csv"), row.names = FALSE)

## ---------------------------------------------------------------- summary --

agg <- function(v) c(median = stats::median(v, na.rm = TRUE),
                     q25 = stats::quantile(v, 0.25, na.rm = TRUE, names = FALSE),
                     q75 = stats::quantile(v, 0.75, na.rm = TRUE, names = FALSE))

summary_rows <- list()
for (model in unique(results$model)) {
  s <- results[results$model == model, ]
  train_agg <- agg(s$train_sec); predict_agg <- agg(s$predict_sec); metric_agg <- agg(s$metric)
  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    dataset = "friedman1", task = "regression", model = model, n_reps = nrow(s),
    train_sec_median = train_agg["median"], train_sec_q25 = train_agg["q25"], train_sec_q75 = train_agg["q75"],
    predict_sec_median = predict_agg["median"], predict_sec_q25 = predict_agg["q25"], predict_sec_q75 = predict_agg["q75"],
    metric_name = "rmse", metric_median = metric_agg["median"],
    metric_q25 = metric_agg["q25"], metric_q75 = metric_agg["q75"]
  )
}
# paired win/loss/tie of fastgbm vs each competitor, per repeat, on train speed and RMSE
# (lower RMSE = win, unlike the survival benchmark's higher-cindex-wins convention)
fg <- results[results$model == "fastgbm", ][order(results[results$model == "fastgbm", ]$rep), ]
for (model in setdiff(unique(results$model), "fastgbm")) {
  other <- results[results$model == model, ][order(results[results$model == model, ]$rep), ]
  m <- merge(fg[, c("rep", "train_sec", "metric")],
            other[, c("rep", "train_sec", "metric")], by = "rep", suffixes = c("_fastgbm", "_other"))
  speed_win <- sum(m$train_sec_fastgbm < m$train_sec_other)
  speed_loss <- sum(m$train_sec_fastgbm > m$train_sec_other)
  speed_tie <- nrow(m) - speed_win - speed_loss
  metric_win <- sum(m$metric_fastgbm < m$metric_other)
  metric_loss <- sum(m$metric_fastgbm > m$metric_other)
  metric_tie <- nrow(m) - metric_win - metric_loss
  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    dataset = "friedman1", task = "regression", model = paste0("fastgbm_vs_", model), n_reps = nrow(m),
    train_sec_median = NA_real_, train_sec_q25 = NA_real_, train_sec_q75 = NA_real_,
    predict_sec_median = NA_real_, predict_sec_q25 = NA_real_, predict_sec_q75 = NA_real_,
    metric_name = paste0("speed_win/loss/tie=", speed_win, "/", speed_loss, "/", speed_tie,
                        "; rmse_win/loss/tie=", metric_win, "/", metric_loss, "/", metric_tie),
    metric_median = NA_real_, metric_q25 = NA_real_, metric_q75 = NA_real_
  )
}
summary_df <- do.call(rbind, summary_rows)
write.csv(summary_df, file.path(out_dir, "regression-benchmark-summary.csv"), row.names = FALSE)

## ------------------------------------------------------------------ PDP shape --
## Fit once on a large low-noise sample, then compare fastgbm's estimated PDP
## for x1/x3/x4/x5 against the *true* partial dependence: since the DGP is
## known, the true PDP of feature j is E_{x_{-j}}[f(x_j, x_{-j})] with x_j
## fixed at each grid value, estimated by Monte Carlo averaging the true
## Friedman1 mean function over the observed rows (same convention pdp()
## itself uses for the fitted model: replace-and-average, not integrate
## analytically).

d_pdp <- friedman1(5000L, seed = SEED0 + 9999L, sd = 0.5)
x_pdp <- d_pdp$x
colnames(x_pdp) <- paste0("x", seq_len(ncol(x_pdp)))
fit_pdp <- fastgbm(x_pdp, y = d_pdp$y, objective = "regression",
                   ntrees = 300L, learning_rate = 0.05, max_depth = MAX_DEPTH,
                   min_node_size = MIN_NODE, threads = 1L, seed = SEED0, verbose = FALSE)

true_mean <- function(x) {
  10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 + 10 * x[, 4] + 5 * x[, 5]
}

true_pdp <- function(x, feature_idx, grid) {
  vapply(grid, function(g) {
    xg <- x
    xg[, feature_idx] <- g
    mean(true_mean(xg))
  }, numeric(1))
}

features <- c(x1 = 1L, x3 = 3L, x4 = 4L, x5 = 5L)
pdp_data <- do.call(rbind, lapply(names(features), function(fname) {
  idx <- features[[fname]]
  fit_pd <- fastgbm::pdp(fit_pdp, fname, data = as.data.frame(x_pdp), grid_resolution = 20L, type = "response")
  true_y <- true_pdp(x_pdp, idx, fit_pd$x)
  data.frame(feature = fname, x = fit_pd$x, fastgbm = fit_pd$yhat, truth = true_y - mean(true_y) + mean(fit_pd$yhat))
}))
write.csv(pdp_data, file.path(out_dir, "regression-pdp-data.csv"), row.names = FALSE)

png(file.path(out_dir, "regression-pdp.png"), width = 900, height = 700, res = 120)
op <- par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
for (fname in names(features)) {
  sub <- pdp_data[pdp_data$feature == fname, ]
  plot(sub$x, sub$truth, type = "l", lwd = 2, col = "grey50", xlab = fname,
      ylab = "partial dependence (recentred)",
      main = fname, ylim = range(c(sub$truth, sub$fastgbm)))
  lines(sub$x, sub$fastgbm, lwd = 2, col = "steelblue", lty = 2)
  legend("topleft", legend = c("true (recentred)", "fastgbm"), col = c("grey50", "steelblue"),
        lty = c(1, 2), lwd = 2, bty = "n", cex = 0.8)
}
par(op)
dev.off()

## ------------------------------------------------------------- session info --

si_file <- file.path(out_dir, "regression-session-info.txt")
con <- file(si_file, "w")
writeLines(c(
  paste("Date:", Sys.time()),
  paste("R version:", R.version.string),
  paste("OS:", Sys.info()["sysname"], Sys.info()["release"]),
  paste("fastgbm version:", as.character(utils::packageVersion("fastgbm"))),
  paste("gbm version:", as.character(utils::packageVersion("gbm"))),
  paste("xgboost version:", as.character(utils::packageVersion("xgboost"))),
  paste("ranger version:", as.character(utils::packageVersion("ranger"))),
  paste("mlbench version:", as.character(utils::packageVersion("mlbench"))),
  paste("Seed base:", SEED0),
  paste("ntrees:", NTREES, "max_depth:", MAX_DEPTH, "learning_rate:", LR, "min_node_size:", MIN_NODE),
  "", "sessionInfo():"
), con)
writeLines(capture.output(sessionInfo()), con)
close(con)

message("Regression benchmark complete. Wrote: ",
       file.path(out_dir, "regression-benchmark-results.csv"), ", ",
       file.path(out_dir, "regression-benchmark-summary.csv"), ", ",
       file.path(out_dir, "regression-pdp.png"), ", ", si_file)
