fastgbm <- function(x, y = NULL, objective = NULL,
                    ntrees = 500L,
                    learning_rate = 0.05,
                    max_depth = 6L,
                    min_node_size = 20L,
                    max_leaves = NULL,
                    max_bins = 255L,
                    subsample = 1,
                    colsample = 1,
                    lambda = 1,
                    gamma = 0,
                    min_child_weight = 1,
                    validation = NULL,
                    early_stopping = NULL,
                    grow_policy = "depthwise",
                    threads = 0L,
                    seed = 1L,
                    verbose = TRUE,
                    ...) {
  if (inherits(x, "formula")) {
    dots <- list(...)
    data <- dots$data
    dots$data <- NULL
    return(do.call(fastgbm_formula, c(list(
      formula = x,
      data = if (is.null(data)) y else data,
      objective = objective
    ), dots, list(
      ntrees = ntrees,
      learning_rate = learning_rate,
      max_depth = max_depth,
      min_node_size = min_node_size,
      max_leaves = max_leaves,
      max_bins = max_bins,
      subsample = subsample,
      colsample = colsample,
      lambda = lambda,
      gamma = gamma,
      min_child_weight = min_child_weight,
      validation = validation,
      early_stopping = early_stopping,
      grow_policy = grow_policy,
      threads = threads,
      seed = seed,
      verbose = verbose
    ))))
  }

  if (is.null(y)) {
    stop("`y` is required for matrix/data frame interfaces.", call. = FALSE)
  }
  if (grow_policy != "depthwise") {
    warning("Only `grow_policy = 'depthwise'` is implemented; using depthwise.", call. = FALSE)
  }
  xmat <- fastgbm_as_matrix(x)
  if (is.null(objective)) {
    objective <- fastgbm_default_objective(fastgbm_validate_y(y, objective))
  }
  y <- fastgbm_prepare_response(y, objective)
  if (!is.null(max_leaves)) {
    warning("`max_leaves` is currently ignored.", call. = FALSE)
  }
  if (!is.null(validation) || !is.null(early_stopping)) {
    warning("Validation-based early stopping is not active in this initial implementation.", call. = FALSE)
  }
  fit <- .Call(
    "fastgbm_fit_cpp",
    xmat,
    as.numeric(y),
    as.character(objective),
    as.integer(ntrees),
    as.numeric(learning_rate),
    as.integer(max_depth),
    as.integer(min_node_size),
    as.integer(max_bins),
    as.numeric(subsample),
    as.numeric(colsample),
    as.numeric(lambda),
    as.numeric(gamma),
    as.numeric(min_child_weight),
    as.integer(seed),
    as.logical(verbose)
  )
  fit$call <- match.call()
  fit$objective <- objective
  fit$ntrees <- length(fit$trees)
  fit$learning_rate <- learning_rate
  fit$max_depth <- max_depth
  fit$min_node_size <- min_node_size
  fit$fitted_raw <- .Call("fastgbm_predict_cpp", fit, xmat, "link")
  fit$fitted <- fastgbm_transform_response(fit$fitted_raw, objective)
  class(fit) <- "fastgbm"
  fit
}
