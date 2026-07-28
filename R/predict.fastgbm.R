predict.fastgbm <- function(object, newdata, type = c("response", "link", "survival"), times = NULL, ...) {
  type <- match.arg(type)
  if (type == "survival") {
    stop("Survival prediction is not implemented in this initial release.", call. = FALSE)
  }
  if (!is.null(object$formula_terms) && is.data.frame(newdata)) {
    mf <- model.frame(object$formula_terms, data = newdata, na.action = na.pass, xlev = object$xlevels)
    newdata <- model.matrix(delete.response(object$formula_terms), data = mf, contrasts.arg = object$contrasts)
  } else {
    newdata <- fastgbm_as_matrix(newdata)
  }
  raw <- .Call("fastgbm_predict_cpp", object, newdata, "link")
  if (type == "link") {
    raw
  } else {
    fastgbm_transform_response(raw, object$objective)
  }
}

coef.fastgbm <- function(object, ...) {
  stop("Tree ensembles do not have ordinary regression coefficients.", call. = FALSE)
}
