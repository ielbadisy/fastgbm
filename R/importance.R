importance <- function(object, ...) {
  imp <- object$feature_importance
  if (is.null(imp)) {
    return(data.frame(feature = character(), gain = numeric()))
  }
  out <- data.frame(
    feature = names(imp),
    gain = as.numeric(imp),
    row.names = NULL
  )
  out[order(out$gain, decreasing = TRUE), , drop = FALSE]
}
