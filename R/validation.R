fastgbm_validate_x <- function(x, allow_dataframe = TRUE) {
  if (is.data.frame(x) && allow_dataframe) {
    return(x)
  }
  if (!is.matrix(x)) {
    stop("`x` must be a matrix or data frame.", call. = FALSE)
  }
  x
}

fastgbm_validate_y <- function(y, objective = NULL) {
  if (is.factor(y)) {
    if (nlevels(y) == 2L) {
      y <- as.numeric(y == levels(y)[2L])
    } else {
      stop("Factor outcomes are only supported for binary classification.", call. = FALSE)
    }
  }
  if (is.logical(y)) {
    y <- as.numeric(y)
  }
  if (!is.numeric(y)) {
    stop("`y` must be numeric, logical, or a binary factor.", call. = FALSE)
  }
  y <- as.numeric(y)
  if (any(!is.finite(y))) {
    stop("`y` must be finite.", call. = FALSE)
  }
  y
}

fastgbm_default_objective <- function(y) {
  u <- sort(unique(y))
  if (length(u) == 2L && all(u %in% c(0, 1))) {
    "binary:logistic"
  } else {
    "reg:squarederror"
  }
}

fastgbm_prepare_matrix <- function(x, formula_terms = NULL, xlevels = NULL, contrasts = NULL) {
  if (inherits(x, "fastgbm.matrix")) {
    return(unclass(x))
  }
  if (is.data.frame(x)) {
    if (!is.null(formula_terms)) {
      mf <- model.frame(formula_terms, data = x, na.action = na.pass, xlev = xlevels)
      mm <- model.matrix(delete.response(formula_terms), data = mf, contrasts.arg = contrasts)
      attr(mm, "terms") <- formula_terms
      attr(mm, "xlevels") <- xlevels
      attr(mm, "contrasts") <- contrasts
      return(mm)
    }
    return(model.matrix(~ . - 1, data = x))
  }
  if (!is.matrix(x)) {
    stop("Could not coerce `x` to a numeric matrix.", call. = FALSE)
  }
  mode(x) <- "double"
  x
}

fastgbm_prepare_response <- function(y, objective) {
  y <- fastgbm_validate_y(y, objective)
  if (objective == "binary:logistic" && !all(y %in% c(0, 1))) {
    stop("Binary logistic objective requires a 0/1 response.", call. = FALSE)
  }
  y
}

fastgbm_transform_response <- function(raw, objective) {
  if (objective == "binary:logistic") {
    1 / (1 + exp(-pmin(pmax(raw, -35), 35)))
  } else {
    raw
  }
}

fastgbm_metric_name <- function(objective) {
  switch(objective,
    "reg:squarederror" = "rmse",
    "binary:logistic" = "logloss",
    "rmse"
  )
}

metrics <- function(object, newdata = NULL, y = NULL, type = c("response", "link")) {
  type <- match.arg(type)
  if (is.null(newdata)) {
    preds <- if (type == "link") object$fitted_raw else object$fitted
  } else {
    preds <- predict(object, newdata, type = type)
  }
  if (is.null(y)) {
    return(preds)
  }
  y <- fastgbm_validate_y(y, object$objective)
  list(
    objective = object$objective,
    metric = fastgbm_metric_name(object$objective),
    value = switch(object$objective,
      "binary:logistic" = logloss(y, preds),
      rmse(y, preds)
    )
  )
}
