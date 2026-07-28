reference_gbm_fit <- function(x, y, ntrees = 10L, learning_rate = 0.1, max_depth = 2L,
                              min_node_size = 2L, lambda = 1, gamma = 0) {
  x <- as.matrix(x)
  y <- as.numeric(y)
  n <- nrow(x)
  pred <- rep(mean(y), n)

  grow_tree <- function(rows, depth) {
    g <- pred[rows] - y[rows]
    h <- rep(1, length(rows))
    leaf_value <- -sum(g) / (sum(h) + lambda)
    if (depth >= max_depth || length(rows) < 2 * min_node_size) {
      return(list(feature = -1L, threshold = -1L, missing_left = TRUE, left = -1L, right = -1L, value = leaf_value))
    }
    best <- list(gain = -Inf)
    for (j in seq_len(ncol(x))) {
      values <- x[rows, j]
      values <- values[is.finite(values)]
      if (length(unique(values)) < 2L) next
      cuts <- sort(unique(values))
      for (thr in cuts[-length(cuts)]) {
        left <- rows[x[rows, j] <= thr | is.na(x[rows, j])]
        right <- setdiff(rows, left)
        if (length(left) < min_node_size || length(right) < min_node_size) next
        GL <- sum(pred[left] - y[left])
        HL <- length(left)
        GR <- sum(pred[right] - y[right])
        HR <- length(right)
        gain <- 0.5 * ((GL^2)/(HL + lambda) + (GR^2)/(HR + lambda) - ((GL + GR)^2)/(HL + HR + lambda)) - gamma
        if (gain > best$gain) {
          best <- list(feature = j - 1L, threshold = thr, missing_left = TRUE, gain = gain,
                       left_rows = left, right_rows = right)
        }
      }
    }
    if (!is.finite(best$gain) || best$gain <= 0) {
      return(list(feature = -1L, threshold = -1L, missing_left = TRUE, left = -1L, right = -1L, value = leaf_value))
    }
    left_node <- grow_tree(best$left_rows, depth + 1L)
    right_node <- grow_tree(best$right_rows, depth + 1L)
    list(
      feature = best$feature,
      threshold = best$threshold,
      missing_left = TRUE,
      left = left_node,
      right = right_node,
      value = leaf_value
    )
  }

  trees <- vector("list", ntrees)
  for (m in seq_len(ntrees)) {
    tree <- grow_tree(seq_len(n), 0L)
    trees[[m]] <- tree
    predict_tree <- function(node, xi) {
      if (node$feature < 0L) return(node$value)
      if (is.na(xi[node$feature + 1L]) || xi[node$feature + 1L] <= node$threshold) {
        predict_tree(node$left, xi)
      } else {
        predict_tree(node$right, xi)
      }
    }
    for (i in seq_len(n)) {
      pred[i] <- pred[i] + learning_rate * predict_tree(tree, x[i, ])
    }
  }
  list(trees = trees, fitted = pred)
}

reference_gbm_predict <- function(model, x, learning_rate = 0.1) {
  x <- as.matrix(x)
  out <- rep(mean(0), nrow(x))
  predict_tree <- function(node, xi) {
    if (node$feature < 0L) return(node$value)
    if (is.na(xi[node$feature + 1L]) || xi[node$feature + 1L] <= node$threshold) {
      predict_tree(node$left, xi)
    } else {
      predict_tree(node$right, xi)
    }
  }
  for (tree in model$trees) {
    for (i in seq_len(nrow(x))) {
      out[i] <- out[i] + learning_rate * predict_tree(tree, x[i, ])
    }
  }
  out
}
