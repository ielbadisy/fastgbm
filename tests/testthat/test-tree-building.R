## Validates that each tree node's stored leaf value equals -G/(H+lambda) for
## the *actual* set of training rows that land in that leaf, computed here
## independently in R by replicating the C++ binning/traversal rules exactly
## (src/fastgbm.cpp: bin_value() and predict_tree_row()). This is a direct
## check on the split-search's internal gradient/Hessian bookkeeping (each
## node's G/H is passed down from its parent's winning split rather than
## recomputed from scratch, for speed -- see NEWS.md), not just on the
## gradient/Hessian *formulas* (that's test-gradients.R). `subsample`/
## `colsample` are fixed at 1 so every row and feature is actually used,
## making the "true" G/H unambiguous to compute by hand.
bin_value_r <- function(v, cuts) {
  if (!is.finite(v)) return(0L)
  sum(cuts <= v) + 1L  # matches bin_value()'s std::upper_bound-based rule exactly
}

leaf_values_match_direct_sums <- function(x, grad, hess, tree, cuts, lambda) {
  p <- ncol(x)
  bins <- vapply(seq_len(p), function(j) vapply(x[, j], bin_value_r, integer(1), cuts = cuts[[j]]), integer(nrow(x)))
  leaf_of <- function(row_bins) {
    node <- 1L  # R is 1-indexed; tree$feature etc. are 0-indexed C++ node ids
    repeat {
      if (tree$feature[node] < 0) return(node)
      f <- tree$feature[node] + 1L
      b <- row_bins[f]
      node <- if (b == 0) {
        if (tree$missing_left[node]) tree$left[node] + 1L else tree$right[node] + 1L
      } else if (b <= tree$threshold[node]) {
        tree$left[node] + 1L
      } else {
        tree$right[node] + 1L
      }
    }
  }
  leaf_ids <- apply(bins, 1, leaf_of)
  for (lf in sort(unique(leaf_ids))) {
    rows_in_leaf <- which(leaf_ids == lf)
    true_value <- -sum(grad[rows_in_leaf]) / (sum(hess[rows_in_leaf]) + lambda)
    expect_equal(tree$value[lf], true_value, tolerance = 1e-10,
                info = sprintf("leaf %d (n=%d)", lf, length(rows_in_leaf)))
  }
  invisible(TRUE)
}

test_that("regression tree leaf values match direct gradient/Hessian sums over their actual rows", {
  set.seed(101)
  n <- 80; p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  fit <- fastgbm(x, y = y, objective = "regression", ntrees = 1L, max_depth = 4L,
                min_node_size = 2L, lambda = 1, subsample = 1, colsample = 1,
                threads = 1L, seed = 1L, verbose = FALSE)

  init_score <- mean(y)
  grad <- rep(init_score, n) - y
  hess <- rep(1, n)
  leaf_values_match_direct_sums(x, grad, hess, fit$trees[[1]], fit$cuts, lambda = 1)
})

test_that("binary tree leaf values match direct gradient/Hessian sums over their actual rows", {
  set.seed(102)
  n <- 80; p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  fit <- fastgbm(x, y = y, objective = "binary", ntrees = 1L, max_depth = 4L,
                min_node_size = 2L, lambda = 1, subsample = 1, colsample = 1,
                threads = 1L, seed = 1L, verbose = FALSE)

  p_bar <- min(max(mean(y), 1e-6), 1 - 1e-6)
  init_score <- log(p_bar / (1 - p_bar))
  prob <- 1 / (1 + exp(-init_score))
  grad <- rep(prob, n) - y
  hess <- rep(pmax(1e-6, prob * (1 - prob)), n)
  leaf_values_match_direct_sums(x, grad, hess, fit$trees[[1]], fit$cuts, lambda = 1)
})

test_that("multi-threaded split search gives identical leaf G/H bookkeeping to single-threaded", {
  set.seed(103)
  n <- 600; p <- 12
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  fit1 <- fastgbm(x, y = y, objective = "regression", ntrees = 5L, max_depth = 5L,
                  min_node_size = 5L, subsample = 1, colsample = 1,
                  threads = 1L, seed = 1L, verbose = FALSE)
  fit2 <- fastgbm(x, y = y, objective = "regression", ntrees = 5L, max_depth = 5L,
                  min_node_size = 5L, subsample = 1, colsample = 1,
                  threads = 4L, seed = 1L, verbose = FALSE)
  expect_identical(fit1$trees, fit2$trees)
})
