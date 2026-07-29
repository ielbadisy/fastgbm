## Next optimization-loop iteration: with early stopping now active, does row
## subsampling (bagging-like) close more of the remaining gap to ranger? The
## earlier regularization probe (diagnose.R) ran WITHOUT early stopping, so the
## interaction between subsample and the now-active early-stopping mechanism
## was untested.

suppressPackageStartupMessages({
  library(survgbm)
  library(biostatlab)
  library(survival)
})

SEED0 <- 20260729L + 1L

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

build_pbc <- function() {
  d <- biostatlab::pbc
  x <- as.matrix(d[, setdiff(names(d), c("time", "status"))]); mode(x) <- "double"
  list(x = x, time = d$time, status = d$status)
}
build_heart_failure <- function() {
  d <- biostatlab::heart_failure
  x <- as.matrix(d[, setdiff(names(d), c("time", "DEATH_EVENT"))]); mode(x) <- "double"
  list(x = x, time = d$time, status = d$DEATH_EVENT)
}
build_breast <- function() {
  d <- biostatlab::breast
  x <- as.matrix(d[, setdiff(names(d), c("time", "status"))]); mode(x) <- "double"
  list(x = x, time = d$time, status = d$status)
}
build_colon_cancer <- function() {
  d <- biostatlab::colon_cancer
  x <- as.matrix(d[, setdiff(names(d), c("time", "status", "etype"))]); mode(x) <- "double"
  list(x = x, time = d$time, status = d$status)
}
to_numeric_matrix <- function(df) {
  cols <- lapply(df, function(col) {
    if (is.numeric(col)) return(as.numeric(col))
    if (is.logical(col)) return(as.numeric(col))
    as.numeric(as.factor(col))
  })
  out <- as.matrix(as.data.frame(cols)); storage.mode(out) <- "double"; colnames(out) <- names(df)
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
  d$scl1[d$scl1 == -1] <- NA; d$scl2[d$scl2 == -1] <- NA; d$smok[d$smok == -1] <- NA
  x <- as.matrix(d[, setdiff(names(d), c("time", "status", "cause", "chd", "cva", "ca", "oth"))])
  mode(x) <- "double"
  list(x = x, time = d$time, status = d$status, has_missing = TRUE)
}

DATASETS <- list(
  pbc = build_pbc(), heart_failure = build_heart_failure(), breast = build_breast(),
  colon_cancer = build_colon_cancer(), crc_mondaca2020 = build_crc_mondaca2020(),
  framingham = build_framingham()
)

SUBSAMPLES <- c(1, 0.8, 0.6)

run_once <- function(x, time, status, seed, subsample, has_missing) {
  sp <- train_test_split(nrow(x), seed)
  x_tr <- x[sp$train, , drop = FALSE]; x_te <- x[sp$test, , drop = FALSE]
  time_tr <- time[sp$train]; time_te <- time[sp$test]
  status_tr <- status[sp$train]; status_te <- status[sp$test]

  es_split <- train_test_split(nrow(x_tr), seed + 1000L, p_train = 0.85)
  x_tr2 <- x_tr[es_split$train, , drop = FALSE]; x_val <- x_tr[es_split$test, , drop = FALSE]
  time_tr2 <- time_tr[es_split$train]; time_val <- time_tr[es_split$test]
  status_tr2 <- status_tr[es_split$train]; status_val <- status_tr[es_split$test]

  fit <- survgbm(x_tr2, time = time_tr2, status = status_tr2, objective = "cox",
                 ntrees = 200L, learning_rate = 0.1, max_depth = 5L, min_node_size = 10L,
                 subsample = subsample, colsample = 0.8,
                 validation = list(x = x_val, time = time_val, status = status_val),
                 early_stopping = 20L, threads = 1L, seed = seed, verbose = FALSE)
  cindex_reverse(time_te, status_te, predict(fit, x_te, type = "link"))
}

for (name in names(DATASETS)) {
  d <- DATASETS[[name]]
  message("== ", name, " ==")
  for (ss in SUBSAMPLES) {
    vals <- sapply(1:5, function(r) run_once(d$x, d$time, d$status, SEED0 + r, ss, isTRUE(d$has_missing)))
    cat(sprintf("  subsample=%.1f: mean cindex=%.4f (sd=%.4f)\n", ss, mean(vals), sd(vals)))
  }
}
