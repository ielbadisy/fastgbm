#' survgbm: Compact Gradient Boosting for Survival Analysis
#'
#' A compact, survival-analysis-only gradient boosting machine with a
#' compiled (Rcpp + RcppParallel) backend.
#'
#' @keywords internal
#' @useDynLib survgbm, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom RcppParallel RcppParallelLibs
#' @importFrom stats approx model.frame model.matrix predict terms delete.response na.omit na.pass as.formula model.response .getXlevels pnorm
#' @importFrom utils head
"_PACKAGE"
