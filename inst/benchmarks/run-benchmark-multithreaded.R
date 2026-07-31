## Multi-threaded head-to-head: every prior benchmark in this directory ran
## every model single-threaded (`threads = 1L` / `num.threads = 1` /
## `nthread = 1`), including the "fastgbm now beats ranger" result in
## run-benchmark-regression.R. That is not the whole story: `ranger` and
## `xgboost` also support multi-threading, and `ranger`'s parallelism model
## (across independent trees, embarrassingly parallel) is structurally
## different from `fastgbm`'s (within a single tree's split search only,
## since boosting is sequential across trees) -- see README's "Scaling
## behavior vs. n, p, and threads" section. This script checks directly
## whether fastgbm's single-threaded win survives once competitors are
## *also* allowed to use multiple threads, rather than assuming it does.
##
## Run at the largest grid point from run-benchmark-scaling.R (n=20000,
## p=100), the regime where threading has the most room to matter; `gbm` is
## excluded (the base `gbm` package has no thread-count control for a single
## `gbm()` fit) and reported once, single-threaded, as a fixed reference
## line only.
##
## Outputs (relative to the package root):
##   inst/benchmarks/multithread-benchmark.csv  -- fastgbm/ranger/xgboost x threads in {1,4,12}, gbm reference
##   inst/benchmarks/multithread-benchmark.png  -- absolute training time vs. threads
##   inst/benchmarks/multithread-speedup.png    -- per-model speedup vs. each model's own threads=1 time
##
## Run from the package root: Rscript inst/benchmarks/run-benchmark-multithreaded.R

suppressPackageStartupMessages({
  library(fastgbm)
  library(mlbench)
  library(gbm)
  library(xgboost)
  library(ranger)
  library(ggplot2)
})

SEED0 <- 20260801L
N <- 20000L
P <- 100L
NTREES <- 100L  # matches run-benchmark-scaling.R's convention, kept low for total runtime
MAX_DEPTH <- 5L
LR <- 0.1
MIN_NODE <- 10L
THREADS_CAP <- 12L
THREADS_GRID <- sort(unique(c(1L, 4L, THREADS_CAP)))
N_REPS <- 5L

timed <- function(expr) {
  gc(FALSE)
  t <- system.time(value <- force(expr))
  unname(t["elapsed"])
}

set.seed(SEED0)
d <- mlbench::mlbench.friedman1(N, sd = 1)
x <- as.matrix(d$x)
if (P > ncol(x)) x <- cbind(x, matrix(runif(N * (P - ncol(x))), N, P - ncol(x)))
colnames(x) <- paste0("x", seq_len(ncol(x)))
y <- as.numeric(d$y)
df <- as.data.frame(x); df$.y <- y
dtr <- xgboost::xgb.DMatrix(x, label = y)

out_dir <- if (dir.exists("inst/benchmarks")) "inst/benchmarks" else "."

rows <- list()
for (th in THREADS_GRID) {
  for (rep in seq_len(N_REPS)) {
    seed <- SEED0 + rep
    t_fastgbm <- timed(fastgbm(x, y = y, objective = "regression", ntrees = NTREES,
                               learning_rate = LR, max_depth = MAX_DEPTH, min_node_size = MIN_NODE,
                               threads = th, seed = seed, verbose = FALSE))
    t_ranger <- timed(ranger::ranger(.y ~ ., data = df, num.trees = NTREES, max.depth = MAX_DEPTH,
                                     min.node.size = MIN_NODE, num.threads = th, seed = seed))
    t_xgboost <- timed(xgboost::xgb.train(
      params = list(objective = "reg:squarederror", eta = LR, max_depth = MAX_DEPTH,
                   min_child_weight = 1, nthread = th, verbosity = 0),
      data = dtr, nrounds = NTREES
    ))
    rows[[length(rows) + 1L]] <- data.frame(threads = th, rep = rep, model = "fastgbm", train_sec = t_fastgbm)
    rows[[length(rows) + 1L]] <- data.frame(threads = th, rep = rep, model = "ranger", train_sec = t_ranger)
    rows[[length(rows) + 1L]] <- data.frame(threads = th, rep = rep, model = "xgboost", train_sec = t_xgboost)
    message(sprintf("threads=%2d rep=%d | fastgbm=%.3fs ranger=%.3fs xgboost=%.3fs",
                    th, rep, t_fastgbm, t_ranger, t_xgboost))
  }
}

## gbm has no thread-count control for a single fit; run once, single-threaded,
## as a fixed reference (repeated N_REPS times at "threads=1" only, so it still
## contributes a comparable median rather than one noisy draw).
for (rep in seq_len(N_REPS)) {
  t_gbm <- timed(gbm::gbm(.y ~ ., data = df, distribution = "gaussian", n.trees = NTREES,
                          interaction.depth = MAX_DEPTH, shrinkage = LR, n.minobsinnode = MIN_NODE,
                          bag.fraction = 1, train.fraction = 1, verbose = FALSE))
  rows[[length(rows) + 1L]] <- data.frame(threads = 1L, rep = rep, model = "gbm (no threading)", train_sec = t_gbm)
}

mt_df <- do.call(rbind, rows)
write.csv(mt_df, file.path(out_dir, "multithread-benchmark.csv"), row.names = FALSE)

agg <- aggregate(train_sec ~ threads + model, data = mt_df, FUN = median)

p <- ggplot2::ggplot(agg[agg$model != "gbm (no threading)", ],
                     ggplot2::aes(x = threads, y = train_sec, color = model)) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 2) +
  ggplot2::geom_hline(
    data = agg[agg$model == "gbm (no threading)", ],
    ggplot2::aes(yintercept = train_sec), linetype = "dashed", color = "grey50"
  ) +
  ggplot2::annotate("text", x = max(THREADS_GRID), y = agg$train_sec[agg$model == "gbm (no threading)"],
                    label = "gbm (no threading)", hjust = 1, vjust = -0.5, size = 3, color = "grey50") +
  ggplot2::labs(
    title = sprintf("Multi-threaded training time, n=%d, p=%d", N, P),
    subtitle = sprintf("median of %d repeats per (model, threads); %d trees, matched hyperparameters", N_REPS, NTREES),
    x = "threads", y = "training time, seconds"
  ) +
  ggplot2::theme_minimal()
ggplot2::ggsave(file.path(out_dir, "multithread-benchmark.png"), plot = p, width = 7, height = 4.5, dpi = 120)

## Second plot: speedup relative to each model's own threads=1 time, so the
## *shape* of scaling (fastgbm's within-tree-only parallelism vs. ranger's
## across-tree, embarrassingly parallel one; xgboost's own histogram-based
## parallelism as a third point of comparison) is visible directly, not
## just each model's absolute position in the first plot.
threaded_models <- setdiff(unique(agg$model), "gbm (no threading)")
speedup_rows <- lapply(threaded_models, function(m) {
  sub <- agg[agg$model == m, ]
  base <- sub$train_sec[sub$threads == 1L]
  data.frame(threads = sub$threads, model = m, speedup = base / sub$train_sec)
})
speedup_df <- do.call(rbind, speedup_rows)

p2 <- ggplot2::ggplot(speedup_df, ggplot2::aes(x = threads, y = speedup, color = model)) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 2) +
  ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  ggplot2::labs(
    title = sprintf("Multi-threaded speedup vs. threads=1, n=%d, p=%d", N, P),
    subtitle = "dashed line = ideal linear speedup; gbm omitted (no thread-count control)",
    x = "threads", y = "speedup vs. threads = 1"
  ) +
  ggplot2::theme_minimal()
ggplot2::ggsave(file.path(out_dir, "multithread-speedup.png"), plot = p2, width = 7, height = 4.5, dpi = 120)

message("Multi-threaded benchmark complete. Wrote: ",
       file.path(out_dir, "multithread-benchmark.csv"), ", ",
       file.path(out_dir, "multithread-benchmark.png"), ", ",
       file.path(out_dir, "multithread-speedup.png"))
