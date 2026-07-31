## Execution-time illustration, on the same Friedman1 dataset used in
## run-benchmark-regression.R, using the `bench` package (`benchr`, originally
## requested, was archived off CRAN on 2025-11-07 and is no longer
## installable in the normal way; `bench` is the actively maintained
## equivalent and gives the same kind of thing -- a boxplot/jitter plot of
## repeated-run timing distributions, not just a single elapsed-time number).
##
## This is a companion illustration to run-benchmark-regression.R's
## systematic 10-repeated-split benchmark: same dataset, same fixed
## train/test split (one repeat, not resampled) and the same hyperparameter
## grid, but timing is measured via `bench::mark()`'s repeated-iteration
## design (`min_iterations`) rather than a single `system.time()` call per
## split, so the plot also shows within-model timing variability.
##
## Outputs (relative to the package root):
##   inst/benchmarks/regression-exectime-bench.csv  -- bench::mark() summary (list-columns dropped)
##   inst/benchmarks/regression-exectime-bench.png  -- bench::mark() timing plot (autoplot)
##
## Run from the package root: Rscript inst/benchmarks/run-benchmark-exectime.R

suppressPackageStartupMessages({
  library(fastgbm)
  library(mlbench)
  library(gbm)
  library(xgboost)
  library(ranger)
  library(bench)
  library(ggplot2)
})

SEED0 <- 20260731L
N <- 2000L
NTREES <- 200L
MAX_DEPTH <- 5L
LR <- 0.1
MIN_NODE <- 10L
MIN_ITERATIONS <- 10L

set.seed(SEED0)
d <- mlbench::mlbench.friedman1(N, sd = 1)
x <- as.matrix(d$x)
y <- as.numeric(d$y)
colnames(x) <- paste0("x", seq_len(ncol(x)))
df <- as.data.frame(x)
df$.y <- y

dtr <- xgboost::xgb.DMatrix(x, label = y)

out_dir <- if (dir.exists("inst/benchmarks")) "inst/benchmarks" else "."

bm <- bench::mark(
  fastgbm = fastgbm(x, y = y, objective = "regression", ntrees = NTREES,
                    learning_rate = LR, max_depth = MAX_DEPTH,
                    min_node_size = MIN_NODE, threads = 1L, seed = SEED0, verbose = FALSE),
  gbm = gbm::gbm(.y ~ ., data = df, distribution = "gaussian", n.trees = NTREES,
                interaction.depth = MAX_DEPTH, shrinkage = LR, n.minobsinnode = MIN_NODE,
                bag.fraction = 1, train.fraction = 1, verbose = FALSE),
  xgboost = xgboost::xgb.train(
    params = list(objective = "reg:squarederror", eta = LR, max_depth = MAX_DEPTH,
                 min_child_weight = 1, nthread = 1, verbosity = 0),
    data = dtr, nrounds = NTREES
  ),
  ranger = ranger::ranger(.y ~ ., data = df, num.trees = NTREES, max.depth = MAX_DEPTH,
                          min.node.size = MIN_NODE, num.threads = 1, seed = SEED0),
  check = FALSE, min_iterations = MIN_ITERATIONS
)

print(bm[, c("expression", "min", "median", "itr/sec", "n_itr")])

## bench_mark objects carry list-columns (memory allocations, gc, per-iteration
## timings) that write.csv() can't serialize; keep only the scalar summary columns.
bm_out <- as.data.frame(bm)
scalar_cols <- names(bm_out)[!vapply(bm_out, is.list, logical(1))]
bm_out <- bm_out[, scalar_cols]
bm_out$expression <- as.character(bm$expression)
bm_out[c("min", "median", "itr/sec", "mem_alloc")] <- lapply(
  bm_out[c("min", "median", "itr/sec", "mem_alloc")],
  function(col) if (inherits(col, "bench_time") || inherits(col, "bench_bytes")) as.numeric(col) else col
)
write.csv(bm_out, file.path(out_dir, "regression-exectime-bench.csv"), row.names = FALSE)

p <- ggplot2::autoplot(bm, type = "jitter") +
  ggplot2::labs(
    title = "Training time on Friedman1 (n = 2000, p = 10)",
    subtitle = sprintf("bench::mark(), %d+ iterations per model, matched hyperparameters", MIN_ITERATIONS)
  )
ggplot2::ggsave(file.path(out_dir, "regression-exectime-bench.png"), plot = p, width = 7, height = 4.5, dpi = 120)

message("Execution-time benchmark complete. Wrote: ",
       file.path(out_dir, "regression-exectime-bench.csv"), ", ",
       file.path(out_dir, "regression-exectime-bench.png"))
