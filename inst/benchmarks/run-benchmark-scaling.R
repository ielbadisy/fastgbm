## Scaling benchmark: how fastgbm's training time (vs. ranger, its closest
## single-threaded competitor on the Friedman1 benchmark) behaves as n and p
## grow, and how much fastgbm's own multi-threaded split search actually
## helps at different data sizes. Complements run-benchmark-exectime.R
## (single fixed n=2000/p=10 snapshot) and run-benchmark-regression.R
## (accuracy-focused, also fixed size) -- this script is about the shape of
## the runtime curve, not a single point estimate.
##
## DGP: Friedman1 (mlbench::mlbench.friedman1(), 10 columns, only x1-x5
## relevant) with extra pure-noise Uniform(0,1) columns appended to reach a
## target p, so "wider" configurations are a realistic stress test (more
## irrelevant features to search over at every split), not a different DGP.
##
## ntrees is reduced to 100 (from the 200 used elsewhere) purely to keep the
## full sweep's total runtime reasonable; all other hyperparameters match
## run-benchmark-exectime.R.
##
## Outputs (relative to the package root):
##   inst/benchmarks/scaling-benchmark-np.csv       -- n x p sweep, fastgbm vs ranger, threads=1
##   inst/benchmarks/scaling-benchmark-np.png       -- plot of the above
##   inst/benchmarks/scaling-benchmark-threads.csv  -- fastgbm thread-count sweep at the largest n x p
##   inst/benchmarks/scaling-benchmark-threads.png  -- plot of the above
##
## Run from the package root: Rscript inst/benchmarks/run-benchmark-scaling.R

suppressPackageStartupMessages({
  library(fastgbm)
  library(mlbench)
  library(ranger)
  library(ggplot2)
})

SEED0 <- 20260801L
NTREES <- 100L
MAX_DEPTH <- 5L
LR <- 0.1
MIN_NODE <- 10L

N_GRID <- c(1000L, 5000L, 20000L)
P_GRID <- c(10L, 50L, 100L)  # includes the 10 Friedman1 columns; extra columns are pure noise
N_CORES <- max(1L, parallel::detectCores(logical = TRUE))
THREADS_GRID <- unique(c(1L, 2L, 4L, N_CORES))

timed <- function(expr) {
  gc(FALSE)
  t <- system.time(value <- force(expr))
  list(value = value, elapsed = unname(t["elapsed"]))
}

make_data <- function(n, p, seed) {
  set.seed(seed)
  d <- mlbench::mlbench.friedman1(n, sd = 1)
  x <- as.matrix(d$x)
  if (p > ncol(x)) {
    extra <- matrix(runif(n * (p - ncol(x))), n, p - ncol(x))
    x <- cbind(x, extra)
  }
  colnames(x) <- paste0("x", seq_len(ncol(x)))
  list(x = x, y = as.numeric(d$y))
}

out_dir <- if (dir.exists("inst/benchmarks")) "inst/benchmarks" else "."

## ------------------------------------------------------------- n x p sweep --

np_rows <- list()
for (n in N_GRID) {
  for (p in P_GRID) {
    d <- make_data(n, p, SEED0 + n + p)
    df <- as.data.frame(d$x); df$.y <- d$y
    message(sprintf("== n=%d, p=%d ==", n, p))

    ft <- timed(fastgbm(d$x, y = d$y, objective = "regression", ntrees = NTREES,
                        learning_rate = LR, max_depth = MAX_DEPTH, min_node_size = MIN_NODE,
                        threads = 1L, seed = SEED0, verbose = FALSE))
    rt <- timed(ranger::ranger(.y ~ ., data = df, num.trees = NTREES, max.depth = MAX_DEPTH,
                               min.node.size = MIN_NODE, num.threads = 1, seed = SEED0))

    np_rows[[length(np_rows) + 1L]] <- data.frame(
      n = n, p = p, model = "fastgbm", train_sec = ft$elapsed
    )
    np_rows[[length(np_rows) + 1L]] <- data.frame(
      n = n, p = p, model = "ranger", train_sec = rt$elapsed
    )
    message(sprintf("  fastgbm: %.3fs, ranger: %.3fs, ratio (fastgbm/ranger): %.2fx",
                    ft$elapsed, rt$elapsed, ft$elapsed / rt$elapsed))
  }
}
np_df <- do.call(rbind, np_rows)
write.csv(np_df, file.path(out_dir, "scaling-benchmark-np.csv"), row.names = FALSE)

p_np <- ggplot2::ggplot(np_df, ggplot2::aes(x = n, y = train_sec, color = model, linetype = model)) +
  ggplot2::geom_line() +
  ggplot2::geom_point() +
  ggplot2::facet_wrap(~ p, labeller = ggplot2::labeller(p = function(x) paste0("p = ", x))) +
  ggplot2::scale_x_log10() +
  ggplot2::scale_y_log10() +
  ggplot2::labs(
    title = "Training time vs. n, by feature count p (single-threaded)",
    subtitle = sprintf("Friedman1 + noise columns, %d trees, matched hyperparameters, log-log axes", NTREES),
    x = "n (log scale)", y = "training time, seconds (log scale)"
  ) +
  ggplot2::theme_minimal()
ggplot2::ggsave(file.path(out_dir, "scaling-benchmark-np.png"), plot = p_np, width = 9, height = 3.5, dpi = 120)

## ------------------------------------------------------------ threads sweep --
## Largest n x p combination in the grid above, where per-node parallelism
## has the best chance to pay off (parallelFor only engages for nodes with
## >= 256 rows and >= 8 sampled features; both are trivially satisfied here
## but not for small/narrow configurations, e.g. n=1000/p=10).

n_big <- max(N_GRID); p_big <- max(P_GRID)
d_big <- make_data(n_big, p_big, SEED0 + n_big + p_big)
message(sprintf("== threads sweep at n=%d, p=%d ==", n_big, p_big))

threads_rows <- list()
for (th in THREADS_GRID) {
  ft <- timed(fastgbm(d_big$x, y = d_big$y, objective = "regression", ntrees = NTREES,
                      learning_rate = LR, max_depth = MAX_DEPTH, min_node_size = MIN_NODE,
                      threads = th, seed = SEED0, verbose = FALSE))
  threads_rows[[length(threads_rows) + 1L]] <- data.frame(threads = th, train_sec = ft$elapsed)
  message(sprintf("  threads=%d: %.3fs", th, ft$elapsed))
}
threads_df <- do.call(rbind, threads_rows)
threads_df$speedup_vs_1thread <- threads_df$train_sec[threads_df$threads == 1L] / threads_df$train_sec
write.csv(threads_df, file.path(out_dir, "scaling-benchmark-threads.csv"), row.names = FALSE)

p_threads <- ggplot2::ggplot(threads_df, ggplot2::aes(x = threads, y = speedup_vs_1thread)) +
  ggplot2::geom_line() +
  ggplot2::geom_point(size = 2) +
  ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  ggplot2::labs(
    title = sprintf("fastgbm multi-threaded speedup at n=%d, p=%d", n_big, p_big),
    subtitle = "dashed line = ideal linear speedup",
    x = "threads", y = "speedup vs. threads = 1"
  ) +
  ggplot2::theme_minimal()
ggplot2::ggsave(file.path(out_dir, "scaling-benchmark-threads.png"), plot = p_threads, width = 6, height = 4.5, dpi = 120)

message("Scaling benchmark complete. Wrote: ",
       file.path(out_dir, "scaling-benchmark-np.csv"), ", ",
       file.path(out_dir, "scaling-benchmark-np.png"), ", ",
       file.path(out_dir, "scaling-benchmark-threads.csv"), ", ",
       file.path(out_dir, "scaling-benchmark-threads.png"))
