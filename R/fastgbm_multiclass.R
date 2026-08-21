#' @noRd
fastgbm_multiclass_fit <- function(xmat, y, ntrees, learning_rate, max_depth,
                                    min_node_size, max_bins, subsample, colsample,
                                    lambda, gamma, min_child_weight, validation,
                                    early_stopping, threads, seed, verbose, call) {
  y_factor <- as.factor(y)
  if (length(y_factor) != nrow(xmat)) {
    stop("`y` must have the same length as `nrow(x)`.", call. = FALSE)
  }
  class_levels <- levels(y_factor)
  K <- length(class_levels)

  validation_y_factor <- NULL
  if (!is.null(validation)) {
    if (is.null(validation$y)) {
      stop("`validation$y` is required for `objective = \"multiclass\"`.", call. = FALSE)
    }
    validation_y_factor <- factor(validation$y, levels = class_levels)
    if (anyNA(validation_y_factor)) {
      stop("`validation$y` contains levels not present in the training response.", call. = FALSE)
    }
  }

  fits <- vector("list", K)
  names(fits) <- class_levels
  for (k in seq_len(K)) {
    lvl <- class_levels[k]
    y_bin <- as.numeric(y_factor == lvl)
    validation_k <- if (is.null(validation)) {
      NULL
    } else {
      list(x = validation$x, y = as.numeric(validation_y_factor == lvl))
    }
    fits[[k]] <- fastgbm(
      xmat,
      y = y_bin, objective = "binary",
      ntrees = ntrees, learning_rate = learning_rate, max_depth = max_depth,
      min_node_size = min_node_size, max_bins = max_bins, subsample = subsample,
      colsample = colsample, lambda = lambda, gamma = gamma,
      min_child_weight = min_child_weight, validation = validation_k,
      early_stopping = if (is.null(validation_k)) NULL else early_stopping,
      threads = threads, seed = seed + k - 1L, verbose = FALSE
    )
  }

  fitted_prob <- fastgbm_multiclass_combine(lapply(fits, function(f) f$fitted), class_levels)

  structure(
    list(
      call = call,
      objective = "multiclass",
      fits = fits,
      levels = class_levels,
      y = y_factor,
      ntrees = vapply(fits, function(f) f$ntrees, integer(1)),
      learning_rate = learning_rate,
      max_depth = max_depth,
      fitted = fitted_prob
    ),
    class = "fastgbm_multiclass"
  )
}

#' @noRd
fastgbm_multiclass_combine <- function(prob_list, class_levels) {
  probs <- do.call(cbind, prob_list)
  probs <- probs / rowSums(probs)
  colnames(probs) <- class_levels
  probs
}

#' Predict from a fitted multiclass fastgbm model
#'
#' `objective = "multiclass"` fits one binary (one-vs-rest) `fastgbm` model
#' per class, all sharing the same hyperparameters. Predicted class
#' probabilities are each class's own binary probability, renormalized to
#' sum to 1 across classes.
#'
#' @param object A fitted `fastgbm_multiclass` object.
#' @param newdata New data to predict on, in the same representation used to fit.
#' @param type `"prob"` (a `nrow(newdata)` x `nlevels(y)` matrix of class
#'   probabilities), or `"class"` (a factor of predicted class labels, the
#'   `argmax` of `"prob"`).
#' @param ... Unused.
#' @return A matrix (`"prob"`) or factor (`"class"`).
#' @export
predict.fastgbm_multiclass <- function(object, newdata, type = c("prob", "class"), ...) {
  type <- match.arg(type)
  if (!is.null(object$formula_terms) && is.data.frame(newdata)) {
    mf <- model.frame(object$formula_terms, data = newdata, na.action = na.pass, xlev = object$xlevels)
    newdata <- model.matrix(delete.response(object$formula_terms), data = mf, contrasts.arg = object$contrasts)
  }
  probs <- fastgbm_multiclass_combine(
    lapply(object$fits, function(f) predict(f, newdata, type = "response")),
    object$levels
  )
  if (type == "prob") {
    return(probs)
  }
  factor(object$levels[max.col(probs, ties.method = "first")], levels = object$levels)
}

#' Print a fitted multiclass fastgbm model
#'
#' @param x A fitted `fastgbm_multiclass` object.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.fastgbm_multiclass <- function(x, ...) {
  cat("fastgbm multiclass model (one-vs-rest)\n")
  cat("  classes:", paste(x$levels, collapse = ", "), "\n")
  cat("  trees per class:", paste(x$ntrees, collapse = ", "), "\n")
  cat("  learning rate:", x$learning_rate, "\n")
  cat("  max depth:", x$max_depth, "\n")
  invisible(x)
}
