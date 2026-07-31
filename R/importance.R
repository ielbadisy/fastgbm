#' Feature importance (total split gain)
#'
#' @param object A fitted `fastgbm` object.
#' @param ... Unused.
#' @return A `data.frame` with columns `feature` and `gain`, sorted by
#'   decreasing gain.
#' @export
importance <- function(object, ...) {
  imp <- object$feature_importance
  if (is.null(imp)) {
    return(data.frame(feature = character(), gain = numeric()))
  }
  # `feature_importance` is a plain numeric vector (no names attribute is set on
  # the C++ side); feature names live separately on the fitted object.
  feature_names <- object$feature_names
  if (is.null(feature_names) || length(feature_names) != length(imp)) {
    feature_names <- names(imp) %||% paste0("V", seq_along(imp))
  }
  out <- data.frame(
    feature = as.character(feature_names),
    gain = as.numeric(imp),
    row.names = NULL
  )
  out[order(out$gain, decreasing = TRUE), , drop = FALSE]
}
