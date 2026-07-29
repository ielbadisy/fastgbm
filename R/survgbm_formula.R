survgbm_formula <- function(formula, data, ..., objective = c("cox", "aft", "pexp")) {
  mf <- model.frame(formula, data = data, na.action = na.pass)
  y <- model.response(mf)
  if (!inherits(y, "Surv")) {
    stop("The response must be a `survival::Surv` object, e.g. `survgbm(Surv(time, status) ~ ., data = dat)`.", call. = FALSE)
  }
  terms <- terms(mf)
  x <- model.matrix(delete.response(terms), data = mf)
  fit <- survgbm(x, y = y, objective = objective, ...)
  fit$formula_terms <- terms
  fit$xlevels <- .getXlevels(terms, mf)
  fit$contrasts <- attr(x, "contrasts")
  fit
}
