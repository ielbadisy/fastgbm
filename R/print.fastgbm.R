print.fastgbm <- function(x, ...) {
  cat("fastgbm model\n")
  cat("  objective:", x$objective, "\n")
  cat("  trees:", x$ntrees, "\n")
  cat("  learning rate:", x$learning_rate, "\n")
  cat("  max depth:", x$max_depth, "\n")
  invisible(x)
}
