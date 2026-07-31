fastgbm_validate_x <- function(x, allow_dataframe = TRUE) {
  if (is.data.frame(x) && allow_dataframe) {
    return(x)
  }
  if (!is.matrix(x)) {
    stop("`x` must be a matrix or data frame.", call. = FALSE)
  }
  x
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
    stop("`time` and `status` are required, or a `survival::Surv` response.", call. = FALSE)
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

fastgbm_transform_response <- function(raw, objective) {
  if (identical(objective, "binary")) {
    1 / (1 + exp(-pmin(pmax(raw, -35), 35)))
  } else if (identical(objective, "regression")) {
    raw
  } else {
    exp(raw)
  }
}

#' @noRd
fastgbm_default_objective <- function(y) {
  if (is.null(y)) {
    stop("`y` is required when `objective` is not supplied and `x` is not a survival input.", call. = FALSE)
  }
  u <- sort(unique(fastgbm_validate_response(y, "regression", check_binary = FALSE)))
  if (length(u) == 2L && all(u %in% c(0, 1))) "binary" else "regression"
}

#' @noRd
fastgbm_validate_response <- function(y, objective, check_binary = TRUE) {
  if (is.factor(y)) {
    if (nlevels(y) != 2L) {
      stop("Factor responses are only supported for binary classification (2 levels).", call. = FALSE)
    }
    y <- as.numeric(y == levels(y)[2L])
  }
  if (is.logical(y)) {
    y <- as.numeric(y)
  }
  if (!is.numeric(y)) {
    stop("`y` must be numeric, logical, or a two-level factor.", call. = FALSE)
  }
  y <- as.numeric(y)
  if (any(!is.finite(y))) {
    stop("`y` must be finite.", call. = FALSE)
  }
  if (check_binary && identical(objective, "binary") && !all(y %in% c(0, 1))) {
    stop("`objective = \"binary\"` requires a 0/1 response.", call. = FALSE)
  }
  y
}

fastgbm_survival_baseline <- function(time, status, lp) {
  # Breslow baseline cumulative hazard: H0(t) = sum_{t_k <= t} d_k / sum_{i: time_i >= t_k} exp(lp_i).
  # The risk set shrinks as t_k grows, so the per-event risk sum is obtained from a single
  # descending cumulative sum of exp(lp) over time-sorted rows (O(n log n)), rather than an
  # incremental accumulator that must actually remove departing subjects.
  o <- order(time)
  time <- time[o]
  status <- status[o]
  et <- exp(pmin(lp[o], 35))
  uniq_event_times <- sort(unique(time[status == 1L]))
  if (!length(uniq_event_times)) {
    return(list(times = numeric(), cumhaz = numeric()))
  }
  risk_sum_desc <- rev(cumsum(rev(et)))
  first_idx <- match(uniq_event_times, time)
  d_k <- vapply(uniq_event_times, function(t) sum(status[time == t]), numeric(1))
  denom <- pmax(risk_sum_desc[first_idx], 1e-12)
  cumhaz <- cumsum(d_k / denom)
  list(times = uniq_event_times, cumhaz = cumhaz)
}

fastgbm_survival_survprob <- function(baseline, lp, times, objective = "cox", sigma = 1) {
  if (!length(times)) {
    return(matrix(numeric(), nrow = length(lp), ncol = 0L))
  }
  if (objective == "cox") {
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

#' Evaluation metric for a fitted fastgbm model
#'
#' Harrell's C-index for survival objectives (`"cox"`, `"aft"`, `"pexp"`),
#' RMSE for `"regression"`, and log loss for `"binary"`.
#'
#' @param object A fitted `fastgbm` object.
#' @param newdata Optional new data.
#' @param y Optional observed outcomes: a `survival::Surv` object for
#'   survival objectives, or a numeric/logical/two-level-factor vector for
#'   `"regression"`/`"binary"`. If omitted, returns the requested predictions
#'   instead of a metric.
#' @param type Prediction type used when `y` is omitted.
#' @return A list with `objective`, `metric`, and `value`, or a numeric
#'   vector of predictions if `y` is omitted.
#' @export
metrics <- function(object, newdata = NULL, y = NULL, type = c("response", "link")) {
  type <- match.arg(type)
  if (is.null(y)) {
    if (is.null(newdata)) {
      return(if (type == "link") object$fitted_raw else object$fitted)
    }
    return(predict(object, newdata, type = type))
  }
  if (object$objective %in% c("regression", "binary")) {
    y <- fastgbm_validate_response(y, object$objective)
    pred <- if (is.null(newdata)) object$fitted else predict(object, newdata, type = "response")
    if (object$objective == "binary") {
      return(list(objective = object$objective, metric = "logloss", value = logloss(y, pred)))
    }
    return(list(objective = object$objective, metric = "rmse", value = rmse(y, pred)))
  }
  # Concordance needs a *risk* score (higher = earlier event). The Cox linear
  # predictor already increases with risk. The AFT linear predictor estimates
  # log(time) instead, which increases with *longer* survival, so it must be
  # negated before it is comparable to a risk score. `type` only controls the
  # scale (link vs response) requested elsewhere; for concordance we always use
  # the (sign-corrected) linear predictor, since monotonic transforms of it do
  # not change the ranking.
  y <- fastgbm_validate_survival(NULL, NULL, y)
  risk_score <- if (is.null(newdata)) object$fitted_raw else predict(object, newdata, type = "link")
  if (object$objective == "aft") {
    risk_score <- -risk_score
  }
  list(
    objective = object$objective,
    metric = "cindex",
    value = survival_cindex(y$time, y$status, risk_score)
  )
}
