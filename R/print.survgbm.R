#' Print a fitted survgbm model
#'
#' @param x A fitted `survgbm` object.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.survgbm <- function(x, ...) {
  cat("survgbm model\n")
  cat("  objective:", x$objective, "\n")
  cat("  trees:", x$ntrees, "\n")
  cat("  learning rate:", x$learning_rate, "\n")
  cat("  max depth:", x$max_depth, "\n")
  if (!is.null(x$stopping_reason) && x$stopping_reason != "disabled") {
    cat("  early stopping:", x$stopping_reason, "\n")
    cat("  best iteration:", x$best_iteration, "of", x$n_trees_grown, "grown\n")
  }
  invisible(x)
}
