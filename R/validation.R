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
  if (inherits(y, "Surv")) {
    return(y)
  }
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

fastgbm_validate_survival <- function(time, status = NULL, y = NULL) {
  if (inherits(y, "Surv")) {
    time <- y[, 1]
    status <- y[, 2]
  }
  if (!is.null(time) && is.null(status) && is.matrix(time) && ncol(time) == 2L) {
    status <- time[, 2]
    time <- time[, 1]
  }
  if (is.null(time) || is.null(status)) {
    stop("Survival objectives require `time` and `status`, or a `Surv` response.", call. = FALSE)
  }
  time <- as.numeric(time)
  status <- as.integer(status)
  if (length(time) != length(status)) {
    stop("`time` and `status` must have the same length.", call. = FALSE)
  }
  if (any(!is.finite(time)) || any(time <= 0)) {
    stop("Survival times must be positive and finite.", call. = FALSE)
  }
  if (length(unique(status)) == 2L && !all(sort(unique(status)) %in% c(0L, 1L))) {
    status <- as.integer(status == max(status))
  }
  if (any(!status %in% c(0L, 1L, FALSE, TRUE))) {
    stop("`status` must be a 0/1 or logical censoring indicator.", call. = FALSE)
  }
  status <- as.integer(status)
  list(time = time, status = status)
}

fastgbm_default_objective <- function(y) {
  if (inherits(y, "Surv")) {
    return("survival:cox")
  }
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
  if (inherits(y, "Surv")) {
    return(y)
  }
  if (objective == "binary:logistic" && !all(y %in% c(0, 1))) {
    stop("Binary logistic objective requires a 0/1 response.", call. = FALSE)
  }
  y
}

fastgbm_transform_response <- function(raw, objective) {
  if (objective == "binary:logistic") {
    1 / (1 + exp(-pmin(pmax(raw, -35), 35)))
  } else if (objective %in% c("survival:cox", "survival:aft")) {
    exp(raw)
  } else {
    raw
  }
}

fastgbm_metric_name <- function(objective) {
  switch(objective,
    "reg:squarederror" = "rmse",
    "binary:logistic" = "logloss",
    "survival:cox" = "cindex",
    "survival:aft" = "cindex",
    "rmse"
  )
}

fastgbm_survival_baseline <- function(time, status, lp) {
  o <- order(time)
  time <- time[o]
  status <- status[o]
  lp <- lp[o]
  et <- exp(pmin(lp, 35))
  uniq_event_times <- sort(unique(time[status == 1L]))
  if (!length(uniq_event_times)) {
    return(list(times = numeric(), cumhaz = numeric()))
  }
  cumhaz <- numeric(length(uniq_event_times))
  risk_sum <- sum(et)
  idx_desc <- order(time, decreasing = TRUE)
  running <- 0.0
  pos <- length(idx_desc)
  seen <- rep(FALSE, length(time))
  # Breslow baseline cumulative hazard on the training sample.
  for (k in seq_along(uniq_event_times)) {
    t_k <- uniq_event_times[k]
    while (pos >= 1L && time[idx_desc[pos]] >= t_k) {
      running <- running + et[idx_desc[pos]]
      seen[idx_desc[pos]] <- TRUE
      pos <- pos - 1L
    }
    d_k <- sum(status[time == t_k])
    denom <- max(running, 1e-12)
    cumhaz[k] <- if (k == 1L) d_k / denom else cumhaz[k - 1L] + d_k / denom
  }
  list(times = uniq_event_times, cumhaz = cumhaz)
}

fastgbm_survival_survprob <- function(baseline, lp, times, objective = "survival:cox", sigma = 1) {
  if (!length(times)) {
    return(matrix(numeric(), nrow = length(lp), ncol = 0L))
  }
  if (objective == "survival:cox") {
    if (is.null(baseline) || !length(baseline$times)) {
      return(matrix(1, nrow = length(lp), ncol = length(times)))
    }
    haz <- approx(baseline$times, baseline$cumhaz, xout = times, method = "constant", rule = 2, f = 0)$y
    out <- outer(exp(pmin(lp, 35)), haz, function(rr, hh) exp(-hh * rr))
    return(out)
  }
  eta <- matrix(rep(lp, each = length(times)), nrow = length(lp))
  z <- (log(times) - eta) / sigma
  1 - pnorm(z)
}

survival_cindex <- function(time, status, score) {
  time <- as.numeric(time)
  status <- as.integer(status)
  score <- as.numeric(score)
  n <- length(time)
  if (n < 2L) {
    return(NA_real_)
  }
  concordant <- 0
  comparable <- 0
  for (i in seq_len(n - 1L)) {
    for (j in (i + 1L):n) {
      if (time[i] == time[j] && status[i] == 0L && status[j] == 0L) next
      if (time[i] < time[j] && status[i] == 1L) {
        comparable <- comparable + 1L
        if (score[i] > score[j]) concordant <- concordant + 1L
        else if (score[i] == score[j]) concordant <- concordant + 0.5
      } else if (time[j] < time[i] && status[j] == 1L) {
        comparable <- comparable + 1L
        if (score[j] > score[i]) concordant <- concordant + 1L
        else if (score[i] == score[j]) concordant <- concordant + 0.5
      }
    }
  }
  if (comparable == 0L) return(NA_real_)
  concordant / comparable
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
  if (inherits(y, "Surv")) {
    y <- fastgbm_validate_survival(NULL, NULL, y)
    return(list(
      objective = object$objective,
      metric = "cindex",
      value = survival_cindex(y$time, y$status, preds)
    ))
  }
  list(
    objective = object$objective,
    metric = fastgbm_metric_name(object$objective),
    value = switch(object$objective,
      "binary:logistic" = logloss(y, preds),
      rmse(y, preds)
    )
  )
}
