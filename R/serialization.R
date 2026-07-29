#' Save a fitted survgbm model
#'
#' @param object A fitted `survgbm` object.
#' @param path Path to write the serialized model to.
#' @return `path`, invisibly.
#' @export
save_survgbm <- function(object, path) {
  saveRDS(object, path)
  invisible(path)
}

#' Load a serialized survgbm model
#'
#' @param path Path to a serialized model, as written by [save_survgbm()].
#' @return A `survgbm` object.
#' @export
load_survgbm <- function(path) {
  readRDS(path)
}
