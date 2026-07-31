#' Save a fitted fastgbm model
#'
#' @param object A fitted `fastgbm` object.
#' @param path Path to write the serialized model to.
#' @return `path`, invisibly.
#' @export
save_fastgbm <- function(object, path) {
  saveRDS(object, path)
  invisible(path)
}

#' Load a serialized fastgbm model
#'
#' @param path Path to a serialized model, as written by [save_fastgbm()].
#' @return A `fastgbm` object.
#' @export
load_fastgbm <- function(path) {
  readRDS(path)
}
