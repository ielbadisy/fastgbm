fastgbm_formula <- function(formula, data, time = NULL, status = NULL, ..., objective = NULL) {
  mf <- model.frame(formula, data = data, na.action = na.pass)
  y <- model.response(mf)
  if (inherits(y, "Surv")) {
    surv <- fastgbm_validate_survival(time, status, y)
    time <- surv$time
    status <- surv$status
    y0 <- NULL
    if (is.null(objective)) objective <- "survival:cox"
  } else {
    y0 <- y
    if (is.factor(y0) && nlevels(y0) == 2L) {
      y0 <- as.numeric(y0 == levels(y0)[2L])
    }
  }
  if (is.null(objective)) {
    objective <- fastgbm_default_objective(as.numeric(y0))
  }
  if (objective %in% c("survival:cox", "survival:aft")) {
    y <- NULL
  } else {
    y <- fastgbm_prepare_response(y0, objective)
  }
  terms <- terms(mf)
  x <- model.matrix(delete.response(terms), data = mf)
  fit <- fastgbm(x, y, time = time, status = status, objective = objective, ...)
  fit$formula_terms <- terms
  fit$xlevels <- .getXlevels(terms, mf)
  fit$contrasts <- attr(x, "contrasts")
  fit
}
