#' Predict from a fitted survgbm model
#'
#' For `objective = "pexp"`, there is no single scalar "linear predictor" the
#' way Cox/AFT have one, since the model is a function of both covariates and
#' time. `type = "link"`/`"response"` instead use the cumulative hazard at the
#' model's full fitted time horizon (`max(object$pexp_cutpoints)`) as a fixed,
#' well-defined risk score (higher = more risk, same ranking convention as
#' Cox), and `type = "survival"` evaluates the hazard-over-time surface
#' directly at the requested `times`.
#'
#' @param object A fitted `survgbm` object.
#' @param newdata New data to predict on, in the same representation used to fit.
#' @param type `"response"` (exp of the linear predictor), `"link"` (raw linear
#'   predictor), or `"survival"` (survival probability at `times`).
#' @param times Required when `type = "survival"`: a vector of times at which
#'   to evaluate the survival function.
#' @param ... Unused.
#' @return A numeric vector (`"response"`/`"link"`) or a matrix of survival
#'   probabilities, `nrow(newdata)` by `length(times)` (`"survival"`).
#' @export
predict.survgbm <- function(object, newdata, type = c("response", "link", "survival"), times = NULL, ...) {
  type <- match.arg(type)
  if (!is.null(object$formula_terms) && is.data.frame(newdata)) {
    mf <- model.frame(object$formula_terms, data = newdata, na.action = na.pass, xlev = object$xlevels)
    newdata <- model.matrix(delete.response(object$formula_terms), data = mf, contrasts.arg = object$contrasts)
  } else {
    newdata <- survgbm_as_matrix(newdata)
  }

  if (object$objective == "pexp") {
    if (type == "survival") {
      if (is.null(times)) {
        stop("`times` is required for survival predictions.", call. = FALSE)
      }
      H <- survgbm_pexp_cumhaz(object, newdata, times)
      return(exp(-H))
    }
    H_final <- survgbm_pexp_cumhaz(object, newdata, max(object$pexp_cutpoints))[, 1]
    return(if (type == "link") log(pmax(H_final, 1e-300)) else H_final)
  }

  raw <- .Call("survgbm_predict_cpp", object, newdata, "link")
  if (type == "survival") {
    if (is.null(times)) {
      stop("`times` is required for survival predictions.", call. = FALSE)
    }
    if (object$objective == "cox") {
      return(survgbm_survival_survprob(object$baseline, raw, times, objective = "cox"))
    }
    return(survgbm_survival_survprob(NULL, raw, times, objective = "aft", sigma = object$survival_sigma %||% 1))
  }
  if (type == "link") {
    raw
  } else {
    survgbm_transform_response(raw, object$objective)
  }
}

#' Tree ensembles have no ordinary regression coefficients
#'
#' @param object A fitted `survgbm` object.
#' @param ... Unused.
#' @return Always errors.
#' @export
coef.survgbm <- function(object, ...) {
  stop("Tree ensembles do not have ordinary regression coefficients.", call. = FALSE)
}
