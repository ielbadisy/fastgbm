#' Summarize a fitted fastgbm model
#'
#' @param object A fitted `fastgbm` object.
#' @param ... Unused.
#' @return A `summary.fastgbm` object.
#' @export
summary.fastgbm <- function(object, ...) {
  top <- head(importance(object), 10)
  structure(
    list(
      objective = object$objective,
      ntrees = object$ntrees,
      learning_rate = object$learning_rate,
      max_depth = object$max_depth,
      feature_importance = top,
      train_metric = object$train_metric,
      stopping_reason = object$stopping_reason,
      best_iteration = object$best_iteration,
      n_trees_grown = object$n_trees_grown
    ),
    class = "summary.fastgbm"
  )
}

#' Print a fastgbm model summary
#'
#' @param x A `summary.fastgbm` object.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.summary.fastgbm <- function(x, ...) {
  if (!is.null(x$stopping_reason) && x$stopping_reason != "disabled") {
    cat("stopping reason:", x$stopping_reason, "\n")
    cat("best iteration:", x$best_iteration, "of", x$n_trees_grown, "grown\n\n")
  }
  print(x$feature_importance)
  invisible(x)
}
