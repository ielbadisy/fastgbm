#' Save a fitted fastgbm model
#'
#' @param object A fitted `fastgbm` object.
#' @param path Path to write the serialized model to.
#' @return `path`, invisibly.
#' @export
savefastgbm <- function(object, path) {
  saveRDS(object, path)
  invisible(path)
}

#' Load a serialized fastgbm model
#'
#' @param path Path to a serialized model, as written by [savefastgbm()].
#' @return A `fastgbm` object.
#' @export
loadfastgbm <- function(path) {
  readRDS(path)
}
