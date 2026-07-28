save_fastgbm <- function(object, path) {
  saveRDS(object, path)
  invisible(path)
}

load_fastgbm <- function(path) {
  readRDS(path)
}
