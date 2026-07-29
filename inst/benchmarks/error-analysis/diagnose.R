## Error-analysis diagnostics for survgbm: where does it lose to ranger, and why?
## Three parts, run per dataset:
##   1. Learning curve: train/test C-index vs. ntrees (checks over/underfitting).
##   2. Per-observation discordance: which subject pairs does survgbm rank wrong
##      that ranger ranks right, and what do those subjects have in common.
##   3. A quick regularization probe: does subsample/colsample < 1 (bagging-like)
##      close any of the gap.
##
## Run from the package root: Rscript inst/benchmarks/error-analysis/diagnose.R

suppressPackageStartupMessages({
  library(survgbm)
  library(biostatlab)
  library(survival)
  library(ranger)
})

SEED <- 20260729L + 1L
NTREES_MAX <- 300L
CHECKPOINTS <- c(5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300)

train_test_split <- function(n, seed, p_train = 0.7) {
  set.seed(seed)
  idx <- sample.int(n)
  n_train <- max(2L, floor(p_train * n))
  list(train = idx[seq_len(n_train)], test = idx[-seq_len(n_train)])
}

cindex_reverse <- function(time, status, pred) {
  survival::concordance(survival::Surv(time, status) ~ pred, reverse = TRUE)$concordance
}

impute_median_mode <- function(x) {
  for (j in seq_len(ncol(x))) {
    col <- x[, j]
    if (anyNA(col)) x[is.na(col), j] <- stats::median(col, na.rm = TRUE)
  }
  x
}

## -------------------------------------------------------------- datasets --

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
  d$scl1[d$scl1 == -1] <- NA
  d$scl2[d$scl2 == -1] <- NA
  d$smok[d$smok == -1] <- NA
  x <- as.matrix(d[, setdiff(names(d), c("time", "status", "cause", "chd", "cva", "ca", "oth"))])
  mode(x) <- "double"
  list(x = x, time = d$time, status = d$status, has_missing = TRUE)
}

DATASETS <- list(
  pbc = build_pbc(),
  heart_failure = build_heart_failure(),
  crc_mondaca2020 = build_crc_mondaca2020(),
  framingham = build_framingham()
)

## ------------------------------------------------------------ learning curve --

learning_curve <- function(x, time, status, seed) {
  sp <- train_test_split(nrow(x), seed)
  x_tr <- x[sp$train, , drop = FALSE]; x_te <- x[sp$test, , drop = FALSE]
  time_tr <- time[sp$train]; time_te <- time[sp$test]
  status_tr <- status[sp$train]; status_te <- status[sp$test]

  fit <- survgbm(x_tr, time = time_tr, status = status_tr, objective = "cox",
                 ntrees = NTREES_MAX, learning_rate = 0.1, max_depth = 5L,
                 min_node_size = 10L, threads = 1L, seed = seed, verbose = FALSE)

  rows <- lapply(CHECKPOINTS, function(k) {
    fit_k <- fit
    fit_k$trees <- fit$trees[seq_len(min(k, length(fit$trees)))]
    p_tr <- predict(fit_k, x_tr, type = "link")
    p_te <- predict(fit_k, x_te, type = "link")
    data.frame(
      ntrees = k,
      cindex_train = cindex_reverse(time_tr, status_tr, p_tr),
      cindex_test = cindex_reverse(time_te, status_te, p_te)
    )
  })
  do.call(rbind, rows)
}

## --------------------------------------------------------- discordance probe --
## Vectorized over each event's comparable set (instead of a full O(n^2) scalar
## double loop) so this stays fast even on framingham's ~1560-row test set.

discordant_pairs <- function(x, time, status, seed, has_missing = FALSE) {
  sp <- train_test_split(nrow(x), seed)
  x_tr <- x[sp$train, , drop = FALSE]; x_te <- x[sp$test, , drop = FALSE]
  time_tr <- time[sp$train]; time_te <- time[sp$test]
  status_tr <- status[sp$train]; status_te <- status[sp$test]
  x_tr_c <- if (has_missing) impute_median_mode(x_tr) else x_tr
  x_te_c <- if (has_missing) impute_median_mode(x_te) else x_te
  df_tr <- as.data.frame(x_tr_c); df_tr$.time <- time_tr; df_tr$.status <- status_tr
  df_te <- as.data.frame(x_te_c)

  fit <- survgbm(x_tr, time = time_tr, status = status_tr, objective = "cox",
                 ntrees = 200L, learning_rate = 0.1, max_depth = 5L,
                 min_node_size = 10L, threads = 1L, seed = seed, verbose = FALSE)
  risk_sg <- predict(fit, x_te, type = "link")

  rf <- ranger::ranger(survival::Surv(.time, .status) ~ ., data = df_tr[, c(colnames(x_tr), ".time", ".status")],
                       num.trees = 200L, max.depth = 5L, min.node.size = 10L, num.threads = 1L, seed = seed)
  pr <- predict(rf, data = df_te)
  risk_rf <- rowSums(pr$chf)

  n <- length(time_te)
  problem_counts <- integer(n)
  n_disagreements <- 0L
  event_idx <- which(status_te == 1L)
  for (i in event_idx) {
    later <- which(time_te > time_te[i])
    if (!length(later)) next
    sg_concordant <- risk_sg[i] > risk_sg[later]
    rf_concordant <- risk_rf[i] > risk_rf[later]
    bad <- (!sg_concordant) & rf_concordant
    nbad <- sum(bad)
    if (nbad > 0L) {
      n_disagreements <- n_disagreements + nbad
      problem_counts[i] <- problem_counts[i] + nbad
      bl <- later[bad]
      tab <- table(bl)
      problem_counts[as.integer(names(tab))] <- problem_counts[as.integer(names(tab))] + as.integer(tab)
    }
  }
  names(problem_counts) <- as.character(seq_len(n))
  list(
    x_te = x_te, time_te = time_te, status_te = status_te,
    risk_sg = risk_sg, risk_rf = risk_rf,
    n_disagreements = n_disagreements,
    problem_subject_counts = sort(problem_counts[problem_counts > 0], decreasing = TRUE)
  )
}

## --------------------------------------------------- regularization probe --

regularization_probe <- function(x, time, status, seed) {
  sp <- train_test_split(nrow(x), seed)
  x_tr <- x[sp$train, , drop = FALSE]; x_te <- x[sp$test, , drop = FALSE]
  time_tr <- time[sp$train]; time_te <- time[sp$test]
  status_tr <- status[sp$train]; status_te <- status[sp$test]

  configs <- list(
    default        = list(subsample = 1,   colsample = 1),
    subsample_only = list(subsample = 0.8, colsample = 1),
    colsample_only = list(subsample = 1,   colsample = 0.8),
    both           = list(subsample = 0.8, colsample = 0.8)
  )
  rows <- list()
  for (nm in names(configs)) {
    cfg <- configs[[nm]]
    cvals <- sapply(1:5, function(rep_seed) {
      fit <- survgbm(x_tr, time = time_tr, status = status_tr, objective = "cox",
                     ntrees = 200L, learning_rate = 0.1, max_depth = 5L, min_node_size = 10L,
                     subsample = cfg$subsample, colsample = cfg$colsample,
                     threads = 1L, seed = seed + rep_seed, verbose = FALSE)
      cindex_reverse(time_te, status_te, predict(fit, x_te, type = "link"))
    })
    rows[[nm]] <- data.frame(config = nm, subsample = cfg$subsample, colsample = cfg$colsample,
                             cindex_mean = mean(cvals), cindex_sd = sd(cvals))
  }
  do.call(rbind, rows)
}

## ------------------------------------------------------------------ driver --

for (name in names(DATASETS)) {
  d <- DATASETS[[name]]
  message("\n================ ", name, " ================")

  message("-- learning curve --")
  lc <- learning_curve(d$x, d$time, d$status, SEED)
  print(lc)

  message("-- regularization probe (mean +/- sd C-index over 5 seeds) --")
  reg <- regularization_probe(d$x, d$time, d$status, SEED)
  print(reg)

  message("-- discordance vs ranger --")
  disc <- discordant_pairs(d$x, d$time, d$status, SEED, has_missing = isTRUE(d$has_missing))
  cat("n test subjects:", length(disc$time_te), "\n")
  cat("n pairs where survgbm wrong but ranger right:", disc$n_disagreements, "\n")
  cat("top 'problem' subjects (row index in test set, count of discordant pairs involved):\n")
  print(head(disc$problem_subject_counts, 10))
  if (length(disc$problem_subject_counts)) {
    top_idx <- as.integer(names(head(disc$problem_subject_counts, 5)))
    cat("their covariates:\n")
    print(data.frame(disc$x_te[top_idx, , drop = FALSE], time = disc$time_te[top_idx],
                     status = disc$status_te[top_idx], risk_sg = disc$risk_sg[top_idx],
                     risk_rf = disc$risk_rf[top_idx]))
    cat("all test-set covariate medians for comparison:\n")
    print(apply(disc$x_te, 2, median, na.rm = TRUE))
  }
}
