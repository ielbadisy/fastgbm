fastgbm <- function(x, y = NULL, time = NULL, status = NULL, objective = NULL,
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
      time = time,
      status = status,
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

  if (grow_policy != "depthwise") {
    warning("Only `grow_policy = 'depthwise'` is implemented; using depthwise.", call. = FALSE)
  }
  xmat <- fastgbm_as_matrix(x)
  if (is.null(objective)) {
    if (!is.null(time) || !is.null(status) || inherits(y, "Surv")) {
      objective <- "survival:cox"
    } else {
      objective <- fastgbm_default_objective(fastgbm_validate_y(y, objective))
    }
  }
  if (is.null(y) && !objective %in% c("survival:cox", "survival:aft")) {
    stop("`y` is required for matrix/data frame interfaces.", call. = FALSE)
  }
  if (objective %in% c("survival:cox", "survival:aft")) {
    surv <- fastgbm_validate_survival(time, status, y)
    time <- surv$time
    status <- surv$status
    y <- NULL
  } else {
    y <- fastgbm_prepare_response(y, objective)
  }
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
    as.numeric(time),
    as.integer(status),
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
  if (objective == "survival:cox") {
    fit$surv_time <- time
    fit$surv_status <- status
    fit$baseline <- fastgbm_survival_baseline(time, status, fit$fitted_raw)
  } else if (objective == "survival:aft") {
    fit$surv_time <- time
    fit$surv_status <- status
    fit$survival_sigma <- 1
  }
  class(fit) <- "fastgbm"
  fit
}
