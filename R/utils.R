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

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

# Internal-only: exposes the compiled gradient/Hessian kernels for testing.
fastgbm_grad_hess <- function(pred, time, status, objective) {
  .Call(
    "fastgbm_grad_hess_cpp",
    as.numeric(pred),
    as.numeric(time),
    as.integer(status),
    as.character(objective)
  )
}
