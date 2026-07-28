fastgbm_formula <- function(formula, data, ..., objective = NULL) {
  mf <- model.frame(formula, data = data, na.action = na.pass)
  y <- model.response(mf)
  y0 <- y
  if (is.factor(y0) && nlevels(y0) == 2L) {
    y0 <- as.numeric(y0 == levels(y0)[2L])
  }
  if (is.null(objective)) {
    objective <- fastgbm_default_objective(as.numeric(y0))
  }
  y <- fastgbm_prepare_response(y0, objective)
  terms <- terms(mf)
  x <- model.matrix(delete.response(terms), data = mf)
  fit <- fastgbm(x, y, objective = objective, ...)
  fit$formula_terms <- terms
  fit$xlevels <- .getXlevels(terms, mf)
  fit$contrasts <- attr(x, "contrasts")
  fit
}
