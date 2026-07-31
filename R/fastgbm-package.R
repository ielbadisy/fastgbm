#' fastgbm: Compact Gradient Boosting for Regression, Classification, and Survival Analysis
#'
#' A compact gradient boosting machine with a compiled (Rcpp + RcppParallel)
#' backend, covering regression, binary classification, and right-censored
#' survival analysis with one interface.
#'
#' @keywords internal
#' @useDynLib fastgbm, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom RcppParallel RcppParallelLibs
#' @importFrom stats approx model.frame model.matrix predict terms delete.response na.omit na.pass as.formula model.response .getXlevels pnorm
#' @importFrom utils head
"_PACKAGE"
