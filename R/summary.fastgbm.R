summary.fastgbm <- function(object, ...) {
  top <- head(importance(object), 10)
  structure(
    list(
      objective = object$objective,
      ntrees = object$ntrees,
      learning_rate = object$learning_rate,
      max_depth = object$max_depth,
      feature_importance = top,
      train_metric = object$train_metric
    ),
    class = "summary.fastgbm"
  )
}

print.summary.fastgbm <- function(x, ...) {
  print(x$feature_importance)
  invisible(x)
}
