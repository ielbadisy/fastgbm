#' Predict from a fitted fastgbm model
#'
#' For `objective = "pexp"`, there is no single scalar "linear predictor" the
#' way Cox/AFT have one, since the model is a function of both covariates and
#' time. `type = "link"`/`"response"` instead use the cumulative hazard at the
#' model's full fitted time horizon (`max(object$pexp_cutpoints)`) as a fixed,
#' well-defined risk score (higher = more risk, same ranking convention as
#' Cox), and `type = "survival"` evaluates the hazard-over-time surface
#' directly at the requested `times`.
#'
#' @param object A fitted `fastgbm` object.
#' @param newdata New data to predict on, in the same representation used to fit.
#' @param type `"response"` (survival: exp of the linear predictor;
#'   `"regression"`: raw prediction; `"binary"`: predicted probability),
#'   `"link"` (raw linear predictor, all objectives), or `"survival"`
#'   (survival probability at `times`; survival objectives only).
#' @param times Required when `type = "survival"`: a vector of times at which
#'   to evaluate the survival function.
#' @param ... Unused.
#' @return A numeric vector (`"response"`/`"link"`) or a matrix of survival
#'   probabilities, `nrow(newdata)` by `length(times)` (`"survival"`).
#' @export
predict.fastgbm <- function(object, newdata, type = c("response", "link", "survival"), times = NULL, ...) {
  type <- match.arg(type)
  if (!is.null(object$formula_terms) && is.data.frame(newdata)) {
    mf <- model.frame(object$formula_terms, data = newdata, na.action = na.pass, xlev = object$xlevels)
    newdata <- model.matrix(delete.response(object$formula_terms), data = mf, contrasts.arg = object$contrasts)
  } else {
    newdata <- fastgbm_as_matrix(newdata)
  }

  if (object$objective %in% c("regression", "binary")) {
    if (type == "survival") {
      stop("`type = \"survival\"` is only available for survival objectives.", call. = FALSE)
    }
    raw <- .Call("fastgbm_predict_cpp", object, newdata, "link")
    return(if (type == "link") raw else fastgbm_transform_response(raw, object$objective))
  }

  if (object$objective == "pexp") {
    if (type == "survival") {
      if (is.null(times)) {
        stop("`times` is required for survival predictions.", call. = FALSE)
      }
      H <- fastgbm_pexp_cumhaz(object, newdata, times)
      return(exp(-H))
    }
    H_final <- fastgbm_pexp_cumhaz(object, newdata, max(object$pexp_cutpoints))[, 1]
    return(if (type == "link") log(pmax(H_final, 1e-300)) else H_final)
  }

  raw <- .Call("fastgbm_predict_cpp", object, newdata, "link")
  if (type == "survival") {
    if (is.null(times)) {
      stop("`times` is required for survival predictions.", call. = FALSE)
    }
    if (object$objective == "cox") {
      return(fastgbm_survival_survprob(object$baseline, raw, times, objective = "cox"))
    }
    return(fastgbm_survival_survprob(NULL, raw, times, objective = "aft", sigma = object$survival_sigma %||% 1))
  }
  if (type == "link") {
    raw
  } else {
    fastgbm_transform_response(raw, object$objective)
  }
}

#' Tree ensembles have no ordinary regression coefficients
#'
#' @param object A fitted `fastgbm` object.
#' @param ... Unused.
#' @return Always errors.
#' @export
coef.fastgbm <- function(object, ...) {
  stop("Tree ensembles do not have ordinary regression coefficients.", call. = FALSE)
}
