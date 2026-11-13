fastgbm_formula <- function(formula, data, ..., objective = NULL) {
  mf <- model.frame(formula, data = data, na.action = na.pass)
  y <- model.response(mf)
  terms <- terms(mf)
  x <- fastgbm_drop_intercept(model.matrix(delete.response(terms), data = mf))
  fit <- fastgbm(x, y = y, objective = objective, ...)
  fit$formula_terms <- terms
  fit$xlevels <- .getXlevels(terms, mf)
  fit$contrasts <- attr(x, "contrasts")
  fit
}

# Trees are invariant to a constant column, so the `(Intercept)` column that
# model.matrix() adds is dropped: it is not a feature.
fastgbm_drop_intercept <- function(x) {
  keep <- colnames(x) != "(Intercept)"
  if (all(keep)) {
    return(x)
  }
  assign <- attr(x, "assign")
  contrasts <- attr(x, "contrasts")
  x <- x[, keep, drop = FALSE]
  if (!is.null(assign)) attr(x, "assign") <- assign[keep]
  if (!is.null(contrasts)) attr(x, "contrasts") <- contrasts
  x
}

# Builds the prediction matrix for a model fitted through the formula
# interface. Models fitted before 0.7.3 kept `(Intercept)` as a feature, so it
# is kept for them.
fastgbm_formula_newdata <- function(object, newdata) {
  mf <- model.frame(object$formula_terms, data = newdata, na.action = na.pass, xlev = object$xlevels)
  mm <- model.matrix(delete.response(object$formula_terms), data = mf, contrasts.arg = object$contrasts)
  feature_names <- object$feature_names %||% object$fits[[1]]$feature_names
  if ("(Intercept)" %in% feature_names) {
    return(mm)
  }
  fastgbm_drop_intercept(mm)
}
