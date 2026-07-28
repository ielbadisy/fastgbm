fastgbm_sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -35), 35)))
}

fastgbm_as_matrix <- function(x) {
  if (is.data.frame(x)) {
    return(model.matrix(~ . - 1, data = x))
  }
  if (is.matrix(x)) {
    mode(x) <- "double"
    return(x)
  }
  stop("Unsupported input type for `x`.", call. = FALSE)
}

fastgbm_model_fields <- function(object) {
  c("objective", "ntrees", "learning_rate", "max_depth", "min_node_size")
}
