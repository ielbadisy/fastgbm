fastgbm_formula <- function(formula, data, ..., objective = NULL) {
  mf <- model.frame(formula, data = data, na.action = na.pass)
  y <- model.response(mf)
  terms <- terms(mf)
  x <- model.matrix(delete.response(terms), data = mf)
  fit <- fastgbm(x, y = y, objective = objective, ...)
  fit$formula_terms <- terms
  fit$xlevels <- .getXlevels(terms, mf)
  fit$contrasts <- attr(x, "contrasts")
  fit
}
