#' Partial dependence for a fitted survgbm model
#'
#' Computes Friedman's partial dependence of the model's prediction on a
#' single feature: for each value on a grid, the feature column is replaced
#' across every row of `data` and the resulting predictions are averaged.
#'
#' Partial dependence is computed on the *linear predictor* (`type = "link"`)
#' by default, i.e. the Cox log-risk score or the AFT log-time location
#' parameter. This is a risk-score (or location-score) partial dependence, not
#' a survival-probability partial dependence; use `predict(object, ..., type =
#' "survival")` directly if a probability-scale summary at specific horizons
#' is needed.
#'
#' @param object a fitted `survgbm` model.
#' @param feature name of the feature (column of `data`) to profile.
#' @param data the data used to compute the partial dependence average;
#'   typically the training data, in the same representation (matrix or
#'   data frame) used to fit `object`.
#' @param grid_resolution number of grid points for numeric features (fewer
#'   are used if the feature has fewer distinct values). Ignored for factor
#'   or character features, where every level is used.
#' @param type `"response"` or `"link"`; defaults to `"link"` (the risk/location
#'   score).
#'
#' @return a `data.frame` with class `survgbm_pdp` and columns `feature`,
#'   `x` (grid value) and `yhat` (average prediction).
#' @export
pdp <- function(object, feature, data, grid_resolution = 20L, type = NULL) {
  if (!inherits(object, "survgbm")) {
    stop("`object` must be a fitted `survgbm` model.", call. = FALSE)
  }
  if (!feature %in% colnames(data)) {
    stop("`feature` (\"", feature, "\") was not found in `colnames(data)`.", call. = FALSE)
  }
  if (is.null(type)) {
    type <- "link"
  }
  type <- match.arg(type, c("response", "link"))

  col <- data[[feature]]
  if (is.factor(col) || is.character(col)) {
    grid <- sort(unique(col))
  } else {
    col_finite <- col[is.finite(col)]
    if (!length(col_finite)) {
      stop("`feature` has no finite values in `data`.", call. = FALSE)
    }
    probs <- seq(0, 1, length.out = max(2L, grid_resolution))
    grid <- sort(unique(stats::quantile(col_finite, probs = probs, names = FALSE, type = 7)))
  }

  yhat <- vapply(grid, function(g) {
    d <- data
    d[[feature]] <- g
    mean(predict(object, d, type = type), na.rm = TRUE)
  }, numeric(1))

  out <- data.frame(feature = feature, x = grid, yhat = yhat, stringsAsFactors = FALSE)
  class(out) <- c("survgbm_pdp", "data.frame")
  attr(out, "pdp_type") <- type
  attr(out, "objective") <- object$objective
  out
}

#' @export
plot.survgbm_pdp <- function(x, ...) {
  ylab <- paste0("Partial dependence (", attr(x, "pdp_type"), ")")
  xlab <- x$feature[1]
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    .data <- ggplot2::.data
    print(
      ggplot2::ggplot(x, ggplot2::aes(x = .data$x, y = .data$yhat)) +
        ggplot2::geom_line() +
        ggplot2::geom_point(size = 0.8) +
        ggplot2::labs(x = xlab, y = ylab) +
        ggplot2::theme_minimal()
    )
    return(invisible(x))
  }
  plot(x$x, x$yhat, type = "b", xlab = xlab, ylab = ylab, ...)
  invisible(x)
}
