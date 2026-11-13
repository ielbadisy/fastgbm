#' Deprecated functions in fastgbm
#'
#' These functions were renamed to glued lowercase names and will be
#' removed in a future release. Use the new names instead.
#'
#' @name fastgbm-deprecated
#' @param ... Arguments passed to the new function.
#' @return The value of the new function.
#' @keywords internal
NULL

#' @rdname fastgbm-deprecated
#' @export
load_fastgbm <- function(...) {
  .Deprecated("loadfastgbm")
  loadfastgbm(...)
}

#' @rdname fastgbm-deprecated
#' @export
save_fastgbm <- function(...) {
  .Deprecated("savefastgbm")
  savefastgbm(...)
}
