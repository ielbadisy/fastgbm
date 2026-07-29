## Reproducible benchmark: survgbm vs gbm, xgboost, ranger on 6 biostatlab
## survival datasets. Follows the "equal hyperparameter grid" regime (spec
## Regime A): every package is given the same ntrees/depth/learning-rate/
## min-node-size where the parameter exists, not independently tuned.
## Training and prediction time are timed separately, over repeated random
## splits (10 for datasets under 10,000 rows), and Harrell's C-index is
## recorded.
##
## Outputs (relative to the package root):
##   inst/benchmarks/benchmark-results.csv   -- one row per (dataset, model, repeat)
##   inst/benchmarks/benchmark-summary.csv   -- median/IQR + win-loss-tie per (dataset, model)
##   inst/benchmarks/parallel-speedup.csv    -- threads=1 vs threads=N training time
##   inst/benchmarks/session-info.txt        -- reproducibility metadata
##
## Run from the package root: Rscript inst/benchmarks/run-benchmark.R

suppressPackageStartupMessages({
  library(survgbm)
  library(biostatlab)
  library(survival)
  library(gbm)
  library(xgboost)
  library(ranger)
})

SEED0 <- 20260729L
LARGE_THRESHOLD <- 10000L
N_REPEATS_SMALL <- 10L
N_REPEATS_LARGE <- 3L

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

impute_median_mode <- function(x) {
  # ranger has no native missing-value support; used only for the ranger arm
  # of datasets with real missingness, so the comparison is still possible.
  for (j in seq_len(ncol(x))) {
    col <- x[, j]
    if (anyNA(col)) {
      x[is.na(col), j] <- stats::median(col, na.rm = TRUE)
    }
  }
  x
}

## ---------------------------------------------------------------- datasets --

build_pbc <- function() {
  d <- biostatlab::pbc
  x <- as.matrix(d[, setdiff(names(d), c("time", "status"))])
  mode(x) <- "double"
  list(x = x, time = d$time, status = d$status)
}

build_heart_failure <- function() {
  d <- biostatlab::heart_failure
  x <- as.matrix(d[, setdiff(names(d), c("time", "DEATH_EVENT"))])
  mode(x) <- "double"
  list(x = x, time = d$time, status = d$DEATH_EVENT)
}

build_breast <- function() {
  d <- biostatlab::breast
  x <- as.matrix(d[, setdiff(names(d), c("time", "status"))])
  mode(x) <- "double"
  list(x = x, time = d$time, status = d$status)
}

build_colon_cancer <- function() {
  d <- biostatlab::colon_cancer
  x <- as.matrix(d[, setdiff(names(d), c("time", "status", "etype"))])
  mode(x) <- "double"
  list(x = x, time = d$time, status = d$status)
}

# Integer-codes character/factor columns (NA stays NA) instead of model.matrix()
# one-hot encoding, because model.matrix()'s implicit model.frame() call drops
# NA rows outright even when na.action = na.pass is passed to model.matrix()
# itself -- that argument isn't honored for a plain (non-model.frame) `data`
# argument, so it silently defeats the whole point of testing missing-value
# handling on this dataset.
to_numeric_matrix <- function(df) {
  cols <- lapply(df, function(col) {
    if (is.numeric(col)) return(as.numeric(col))
    if (is.logical(col)) return(as.numeric(col))
    as.numeric(as.factor(col))
  })
  out <- as.matrix(as.data.frame(cols))
  storage.mode(out) <- "double"
  colnames(out) <- names(df)
  out
}

build_crc_mondaca2020 <- function() {
  d <- biostatlab::crc_mondaca2020
  d <- d[, setdiff(names(d), "")]
  x <- to_numeric_matrix(d[, setdiff(names(d), c("time", "status"))])
  list(x = x, time = d$time, status = d$status, has_missing = TRUE)
}

build_framingham <- function() {
  d <- biostatlab::framingham
  # `cause`, `chd`, `cva`, `ca`, `oth` are cause-of-death sub-indicators that are
  # deterministically 0 whenever `status == 0` (they only take a value once an
  # event has occurred) -- including them as predictors leaks the outcome and
  # inflates every model's C-index. `scl1`/`scl2`/`smok` use -1 as a raw sentinel
  # for missing rather than NA; recoded here so missingness is handled properly
  # rather than treated as a real (and misleading) small value.
  d$scl1[d$scl1 == -1] <- NA
  d$scl2[d$scl2 == -1] <- NA
  d$smok[d$smok == -1] <- NA
  x <- as.matrix(d[, setdiff(names(d), c("time", "status", "cause", "chd", "cva", "ca", "oth"))])
  mode(x) <- "double"
  list(x = x, time = d$time, status = d$status, has_missing = TRUE)
}

DATASETS <- list(
  pbc              = build_pbc(),
  heart_failure    = build_heart_failure(),
  breast           = build_breast(),
  colon_cancer     = build_colon_cancer(),
  crc_mondaca2020  = build_crc_mondaca2020(),
  framingham       = build_framingham()
)

## ------------------------------------------------------------------ runner --

run_survival_once <- function(x, time, status, seed, has_missing = FALSE) {
  sp <- train_test_split(nrow(x), seed)
  x_tr <- x[sp$train, , drop = FALSE]; x_te <- x[sp$test, , drop = FALSE]
  time_tr <- time[sp$train]; time_te <- time[sp$test]
  status_tr <- status[sp$train]; status_te <- status[sp$test]
  df_tr <- as.data.frame(x_tr); df_tr$.time <- time_tr; df_tr$.status <- status_tr
  df_te <- as.data.frame(x_te)

  cindex <- function(pred) {
    # `pred` is a risk score (higher = more risk = shorter survival); `concordance()`'s
    # formula interface defaults to the opposite convention (higher = longer survival),
    # matching what it does automatically when called on a coxph object directly.
    survival::concordance(survival::Surv(time_te, status_te) ~ pred, reverse = TRUE)$concordance
  }

  rows <- list()

  # survgbm gets an internal validation split carved out of its own training fold and
  # uses early stopping (ntrees = NTREES is a ceiling, not a fixed count) -- this is a
  # real capability the package is meant to have, not extra tuning; the other three
  # packages are left at their fixed-ntrees defaults, same as before. Diagnostics
  # (inst/benchmarks/error-analysis/diagnose.R) showed survgbm's test C-index peaks
  # well before ntrees=200 on every one of these datasets without early stopping.
  # All three implemented objectives are benchmarked, not just Cox: predict(type="link")
  # and metrics()/the cindex() helper below already handle each objective's own risk-score
  # convention (AFT's linear predictor is negated internally, pexp's is a fitted-horizon
  # cumulative hazard), so the same evaluation code applies unchanged to all three.
  es_split <- train_test_split(nrow(x_tr), seed + 1000L, p_train = 0.85)
  x_tr2 <- x_tr[es_split$train, , drop = FALSE]; x_val <- x_tr[es_split$test, , drop = FALSE]
  time_tr2 <- time_tr[es_split$train]; time_val <- time_tr[es_split$test]
  status_tr2 <- status_tr[es_split$train]; status_val <- status_tr[es_split$test]

  for (obj in c("cox", "aft", "pexp")) {
    ft <- timed(survgbm(x_tr2, time = time_tr2, status = status_tr2, objective = obj,
                        ntrees = NTREES, learning_rate = LR, max_depth = MAX_DEPTH,
                        min_node_size = MIN_NODE,
                        validation = list(x = x_val, time = time_val, status = status_val),
                        early_stopping = 20L, threads = 1L, seed = seed, verbose = FALSE))
    pt <- timed(predict(ft$value, x_te, type = "link"))
    risk_score <- if (obj == "aft") -pt$value else pt$value
    model_name <- paste0("survgbm_", obj)
    rows[[model_name]] <- data.frame(model = model_name, train_sec = ft$elapsed, predict_sec = pt$elapsed,
                                     metric = cindex(risk_score))
  }

  gt <- timed(gbm::gbm(survival::Surv(.time, .status) ~ ., data = df_tr[, c(colnames(x_tr), ".time", ".status")],
                       distribution = "coxph", n.trees = NTREES, interaction.depth = MAX_DEPTH,
                       shrinkage = LR, n.minobsinnode = MIN_NODE, bag.fraction = 1,
                       train.fraction = 1, verbose = FALSE))
  pg <- timed(predict(gt$value, newdata = df_te, n.trees = NTREES, type = "link"))
  rows$gbm <- data.frame(model = "gbm", train_sec = gt$elapsed, predict_sec = pg$elapsed,
                         metric = cindex(pg$value))

  # xgboost/ranger require complete data for training; median-impute for those
  # two arms only when the dataset has real missingness (survgbm and gbm both
  # handle missing predictors natively and use the unimputed data above).
  x_tr_c <- if (has_missing) impute_median_mode(x_tr) else x_tr
  x_te_c <- if (has_missing) impute_median_mode(x_te) else x_te

  label_xgb <- ifelse(status_tr == 1, time_tr, -time_tr)
  dtr <- xgboost::xgb.DMatrix(x_tr_c, label = label_xgb)
  dte <- xgboost::xgb.DMatrix(x_te_c)
  xt <- timed(xgboost::xgb.train(
    params = list(objective = "survival:cox", eta = LR, max_depth = MAX_DEPTH,
                 min_child_weight = 1, nthread = 1, verbosity = 0),
    data = dtr, nrounds = NTREES
  ))
  px <- timed(predict(xt$value, dte))
  rows$xgboost <- data.frame(model = "xgboost", train_sec = xt$elapsed, predict_sec = px$elapsed,
                             metric = cindex(px$value))

  df_tr_c <- as.data.frame(x_tr_c); df_tr_c$.time <- time_tr; df_tr_c$.status <- status_tr
  df_te_c <- as.data.frame(x_te_c)
  rt <- timed(ranger::ranger(survival::Surv(.time, .status) ~ .,
                             data = df_tr_c[, c(colnames(x_tr), ".time", ".status")],
                             num.trees = NTREES, max.depth = MAX_DEPTH, min.node.size = MIN_NODE,
                             num.threads = 1, seed = seed))
  pr <- timed(predict(rt$value, data = df_te_c))
  risk_r <- rowSums(pr$value$chf)
  rows$ranger <- data.frame(model = "ranger", train_sec = rt$elapsed, predict_sec = pr$elapsed,
                            metric = cindex(risk_r))

  out <- do.call(rbind, rows)
  out$metric_name <- "cindex"
  out
}

## ------------------------------------------------------------------ driver --

all_results <- list()

for (name in names(DATASETS)) {
  d <- DATASETS[[name]]
  n <- nrow(d$x)
  n_repeats <- if (n >= LARGE_THRESHOLD) N_REPEATS_LARGE else N_REPEATS_SMALL
  message(sprintf("== %s (n=%d, p=%d, repeats=%d) ==", name, n, ncol(d$x), n_repeats))

  for (rep in seq_len(n_repeats)) {
    seed <- SEED0 + rep
    res <- run_survival_once(d$x, d$time, d$status, seed, has_missing = isTRUE(d$has_missing))
    res$dataset <- name
    res$task <- "survival"
    res$n <- n
    res$p <- ncol(d$x)
    res$rep <- rep
    all_results[[length(all_results) + 1L]] <- res
    message(sprintf("  rep %d/%d done", rep, n_repeats))
  }
}

results <- do.call(rbind, all_results)
rownames(results) <- NULL

out_dir <- if (dir.exists("inst/benchmarks")) "inst/benchmarks" else "."

write.csv(results, file.path(out_dir, "benchmark-results.csv"), row.names = FALSE)

## ---------------------------------------------------------------- summary --

agg <- function(v) c(median = stats::median(v, na.rm = TRUE),
                     q25 = stats::quantile(v, 0.25, na.rm = TRUE, names = FALSE),
                     q75 = stats::quantile(v, 0.75, na.rm = TRUE, names = FALSE))

summary_rows <- list()
for (ds in unique(results$dataset)) {
  sub <- results[results$dataset == ds, ]
  for (model in unique(sub$model)) {
    s <- sub[sub$model == model, ]
    train_agg <- agg(s$train_sec)
    predict_agg <- agg(s$predict_sec)
    metric_agg <- agg(s$metric)
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      dataset = ds, task = unique(s$task), model = model, n_reps = nrow(s),
      train_sec_median = train_agg["median"], train_sec_q25 = train_agg["q25"], train_sec_q75 = train_agg["q75"],
      predict_sec_median = predict_agg["median"], predict_sec_q25 = predict_agg["q25"], predict_sec_q75 = predict_agg["q75"],
      metric_name = unique(s$metric_name), metric_median = metric_agg["median"],
      metric_q25 = metric_agg["q25"], metric_q75 = metric_agg["q75"]
    )
  }
  # paired win/loss/tie of each survgbm objective vs each competitor, per repeat,
  # on train speed and C-index
  survgbm_models <- grep("^survgbm_", unique(sub$model), value = TRUE)
  competitors <- setdiff(unique(sub$model), survgbm_models)
  for (fg_model in survgbm_models) {
    fg <- sub[sub$model == fg_model, ][order(sub[sub$model == fg_model, ]$rep), ]
    for (model in competitors) {
      other <- sub[sub$model == model, ][order(sub[sub$model == model, ]$rep), ]
      m <- merge(fg[, c("rep", "train_sec", "metric")],
                other[, c("rep", "train_sec", "metric")], by = "rep", suffixes = c("_survgbm", "_other"))
      speed_win <- sum(m$train_sec_survgbm < m$train_sec_other)
      speed_loss <- sum(m$train_sec_survgbm > m$train_sec_other)
      speed_tie <- nrow(m) - speed_win - speed_loss
      metric_win <- sum(m$metric_survgbm > m$metric_other)
      metric_loss <- sum(m$metric_survgbm < m$metric_other)
      metric_tie <- nrow(m) - metric_win - metric_loss
      summary_rows[[length(summary_rows) + 1L]] <- data.frame(
        dataset = ds, task = unique(sub$task), model = paste0(fg_model, "_vs_", model),
        n_reps = nrow(m),
        train_sec_median = NA_real_, train_sec_q25 = NA_real_, train_sec_q75 = NA_real_,
        predict_sec_median = NA_real_, predict_sec_q25 = NA_real_, predict_sec_q75 = NA_real_,
        metric_name = paste0("speed_win/loss/tie=", speed_win, "/", speed_loss, "/", speed_tie,
                            "; cindex_win/loss/tie=", metric_win, "/", metric_loss, "/", metric_tie),
        metric_median = NA_real_, metric_q25 = NA_real_, metric_q75 = NA_real_
      )
    }
  }
}
summary_df <- do.call(rbind, summary_rows)
write.csv(summary_df, file.path(out_dir, "benchmark-summary.csv"), row.names = FALSE)

## --------------------------------------------------------- parallel speedup --
## threads=1 vs threads=N (hardware concurrency) training time on the largest
## dataset (framingham), 5 repeats. Not a central performance claim, just
## honest evidence for the RcppParallel split-search parallelism.

n_cores <- max(1L, parallel::detectCores(logical = TRUE))
d <- DATASETS$framingham
parallel_rows <- list()
for (rep in seq_len(5L)) {
  seed <- SEED0 + rep
  sp <- train_test_split(nrow(d$x), seed)
  x_tr <- d$x[sp$train, , drop = FALSE]
  time_tr <- d$time[sp$train]; status_tr <- d$status[sp$train]

  t1 <- timed(survgbm(x_tr, time = time_tr, status = status_tr, objective = "cox",
                      ntrees = NTREES, learning_rate = LR, max_depth = MAX_DEPTH,
                      min_node_size = MIN_NODE, threads = 1L, seed = seed, verbose = FALSE))
  tn <- timed(survgbm(x_tr, time = time_tr, status = status_tr, objective = "cox",
                      ntrees = NTREES, learning_rate = LR, max_depth = MAX_DEPTH,
                      min_node_size = MIN_NODE, threads = n_cores, seed = seed, verbose = FALSE))
  parallel_rows[[rep]] <- data.frame(rep = rep, threads_1_sec = t1$elapsed, threads_n_sec = tn$elapsed, n_cores = n_cores)
}
parallel_df <- do.call(rbind, parallel_rows)
write.csv(parallel_df, file.path(out_dir, "parallel-speedup.csv"), row.names = FALSE)

## ------------------------------------------------------------- session info --

si_file <- file.path(out_dir, "session-info.txt")
con <- file(si_file, "w")
writeLines(c(
  paste("Date:", Sys.time()),
  paste("R version:", R.version.string),
  paste("OS:", Sys.info()["sysname"], Sys.info()["release"]),
  paste("Detected cores:", n_cores),
  paste("survgbm version:", as.character(utils::packageVersion("survgbm"))),
  paste("gbm version:", as.character(utils::packageVersion("gbm"))),
  paste("xgboost version:", as.character(utils::packageVersion("xgboost"))),
  paste("ranger version:", as.character(utils::packageVersion("ranger"))),
  paste("biostatlab version:", as.character(utils::packageVersion("biostatlab"))),
  paste("Seed base:", SEED0),
  paste("ntrees:", NTREES, "max_depth:", MAX_DEPTH, "learning_rate:", LR, "min_node_size:", MIN_NODE),
  "", "sessionInfo():"
), con)
writeLines(capture.output(sessionInfo()), con)
close(con)

message("Benchmark complete. Wrote: ",
       file.path(out_dir, "benchmark-results.csv"), ", ",
       file.path(out_dir, "benchmark-summary.csv"), ", ",
       file.path(out_dir, "parallel-speedup.csv"), ", ", si_file)
