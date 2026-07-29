#' Fit a compact gradient boosting model for survival analysis
#'
#' Histogram-based gradient boosting for right-censored survival data, with a
#' Cox (Breslow ties), AFT (normal location-scale), or piecewise-exponential
#' objective and native missing-value routing.
#'
#' @param x Feature matrix, data frame, or a `Surv(time, status) ~ .` formula.
#' @param time Survival time. Ignored if `y` is a `survival::Surv` object or
#'   if `x` is a formula.
#' @param status Event indicator (1 = event, 0 = censored). Ignored if `y` is
#'   a `survival::Surv` object or if `x` is a formula.
#' @param y Optional `survival::Surv(time, status)` object, as an alternative
#'   to passing `time`/`status` separately.
#' @param objective `"cox"` (default), `"aft"`, or `"pexp"` (piecewise
#'   exponential: the ensemble models the log hazard rate jointly over
#'   covariates and time, via a person-time expansion; see [survgbm_pexp_cutpoints()]
#'   and `vignette("survival", package = "survgbm")`).
#' @param ntrees Number of boosting rounds (an upper bound when `early_stopping`
#'   is used). Defaults to `200`, matched to `learning_rate`/`max_depth`/
#'   `min_node_size` below in the benchmark diagnostics
#'   (`inst/benchmarks/error-analysis/`). **Without `validation`/
#'   `early_stopping`, training all `ntrees` rounds reliably overfits** on
#'   small-to-medium survival datasets -- test-set C-index was found to peak
#'   between 10 and 100 rounds and then degrade with further training on
#'   every one of the 6 benchmark datasets. Supplying `validation`/
#'   `early_stopping` is strongly recommended for any dataset where held-out
#'   performance matters.
#' @param learning_rate Shrinkage parameter. Defaults to `0.1`.
#' @param max_depth Maximum tree depth. Defaults to `5`.
#' @param min_node_size Minimum rows in a node before splitting. Defaults to `10`.
#' @param max_leaves Unused placeholder for compatibility.
#' @param max_bins Maximum number of histogram bins.
#' @param subsample Row subsampling fraction. Defaults to `0.8`; a repeat of
#'   the regularization diagnostic with early stopping active showed this
#'   gives a further, mostly-positive C-index gain over `subsample = 1` on
#'   top of the `colsample` default below.
#' @param colsample Feature subsampling *fraction, resampled at every node*
#'   (like `ranger`'s `mtry`, not xgboost's per-tree `colsample_bytree`) --
#'   diagnostics found per-node resampling decorrelates sibling splits more
#'   effectively than per-tree sampling, which is a large part of why random
#'   forests have lower variance than boosting on the same data. Defaults to
#'   `0.8`; diagnostics on the benchmark datasets showed this gives a small,
#'   consistently non-negative gain in C-index over `colsample = 1`, though a
#'   much smaller value close to `sqrt(p)/p` (matching `ranger`'s actual
#'   `mtry` default) did better still on higher-`p` datasets and worse on
#'   low-`p` ones -- there is no universally optimal single value found so far.
#' @param lambda L2 regularization.
#' @param gamma Split penalty.
#' @param min_child_weight Minimum child Hessian sum.
#' @param validation A list with `x` (in the same matrix representation as
#'   the training data), and either `time`/`status` or a `survival::Surv`
#'   `y`, used for early stopping. Must be supplied together with
#'   `early_stopping`.
#' @param early_stopping Number of boosting rounds without validation-loss
#'   improvement before stopping. Must be supplied together with
#'   `validation`. When active, `fit$trees` is truncated to the
#'   best-validation-loss iteration after training, so predictions use the
#'   best model by default; `fit$n_trees_grown`, `fit$validation_history`,
#'   `fit$best_iteration`, and `fit$stopping_reason` record the full run.
#' @param pexp_bins Number of piecewise-exponential time intervals, used only
#'   when `objective = "pexp"`. Cutpoints are quantiles of the observed event
#'   times (see [survgbm_pexp_cutpoints()]).
#' @param grow_policy Tree growth policy; only `"depthwise"` is implemented.
#' @param threads Number of threads for the RcppParallel split search
#'   (`0` = automatic).
#' @param seed Random seed.
#' @param verbose Whether to print progress.
#' @param ... Additional arguments (e.g. `data` for the formula interface).
#'
#' @return A fitted `survgbm` object.
#' @export
survgbm <- function(x, time = NULL, status = NULL, y = NULL, objective = c("cox", "aft", "pexp"),
                    ntrees = 200L,
                    learning_rate = 0.1,
                    max_depth = 5L,
                    min_node_size = 10L,
                    max_leaves = NULL,
                    max_bins = 255L,
                    subsample = 0.8,
                    colsample = 0.8,
                    lambda = 1,
                    gamma = 0,
                    min_child_weight = 1,
                    validation = NULL,
                    early_stopping = NULL,
                    pexp_bins = 10L,
                    grow_policy = "depthwise",
                    threads = 0L,
                    seed = 1L,
                    verbose = TRUE,
                    ...) {
  if (inherits(x, "formula")) {
    dots <- list(...)
    if (is.null(dots$data)) {
      stop("`data` is required when `x` is a formula, e.g. `survgbm(Surv(time, status) ~ ., data = dat)`.", call. = FALSE)
    }
    return(do.call(survgbm_formula, c(list(
      formula = x,
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
      pexp_bins = pexp_bins,
      grow_policy = grow_policy,
      threads = threads,
      seed = seed,
      verbose = verbose
    ))))
  }

  objective <- match.arg(objective)
  if (grow_policy != "depthwise") {
    warning("Only `grow_policy = 'depthwise'` is implemented; using depthwise.", call. = FALSE)
  }
  xmat <- survgbm_as_matrix(x)
  surv <- survgbm_validate_survival(time, status, y)
  time <- surv$time
  status <- surv$status
  if (length(time) != nrow(xmat)) {
    stop("`time`/`status` must have the same length as `nrow(x)`.", call. = FALSE)
  }
  if (!is.null(max_leaves)) {
    warning("`max_leaves` is currently ignored.", call. = FALSE)
  }

  is_pexp <- objective == "pexp"
  pexp_cutpoints <- NULL
  if (is_pexp) {
    pexp_cutpoints <- survgbm_pexp_cutpoints(time, status, bins = pexp_bins)
    expanded <- survgbm_expand_person_time(xmat, time, status, pexp_cutpoints)
    fit_xmat <- expanded$x
    fit_time <- expanded$exposure    # reuses the "time" C++ argument slot as exposure
    fit_status <- expanded$event     # reuses the "status" C++ argument slot as event
  } else {
    fit_xmat <- xmat
    fit_time <- time
    fit_status <- status
  }

  has_validation <- !is.null(validation) || !is.null(early_stopping)
  x_valid_mat <- matrix(numeric(0), nrow = 0, ncol = ncol(fit_xmat))
  time_valid <- numeric(0)
  status_valid <- integer(0)
  early_stopping_int <- 0L
  if (has_validation) {
    if (is.null(validation) || is.null(early_stopping)) {
      stop("`validation` and `early_stopping` must be supplied together.", call. = FALSE)
    }
    if (is.null(validation$x)) {
      stop("`validation` must include `x`, in the same matrix representation as the training data.", call. = FALSE)
    }
    x_valid_raw <- survgbm_as_matrix(validation$x)
    if (ncol(x_valid_raw) != ncol(xmat)) {
      stop("`validation$x` must have the same number of columns as the training data.", call. = FALSE)
    }
    surv_valid <- survgbm_validate_survival(validation$time, validation$status, validation$y)
    time_valid_raw <- surv_valid$time
    status_valid_raw <- surv_valid$status
    if (nrow(x_valid_raw) != length(time_valid_raw)) {
      stop("`validation$x` and `validation$time`/`validation$status` must have the same number of rows.", call. = FALSE)
    }
    if (is_pexp) {
      expanded_valid <- survgbm_expand_person_time(x_valid_raw, time_valid_raw, status_valid_raw, pexp_cutpoints)
      x_valid_mat <- expanded_valid$x
      time_valid <- expanded_valid$exposure
      status_valid <- expanded_valid$event
    } else {
      x_valid_mat <- x_valid_raw
      time_valid <- time_valid_raw
      status_valid <- status_valid_raw
    }
    early_stopping_int <- as.integer(early_stopping)
    if (length(early_stopping_int) != 1L || is.na(early_stopping_int) || early_stopping_int <= 0L) {
      stop("`early_stopping` must be a single positive integer.", call. = FALSE)
    }
  }

  fit <- .Call(
    "survgbm_fit_cpp",
    fit_xmat,
    as.numeric(fit_time),
    as.integer(fit_status),
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
    as.integer(threads),
    as.logical(verbose),
    x_valid_mat,
    as.numeric(time_valid),
    as.integer(status_valid),
    early_stopping_int
  )
  fit$call <- match.call()
  fit$objective <- objective
  fit$learning_rate <- learning_rate
  fit$max_depth <- max_depth
  fit$min_node_size <- min_node_size
  fit$n_trees_grown <- length(fit$trees)
  if (has_validation) {
    fit$trees <- fit$trees[seq_len(fit$best_iteration)]
  }
  fit$ntrees <- length(fit$trees)
  fit$surv_time <- time
  fit$surv_status <- status

  if (is_pexp) {
    fit$pexp_cutpoints <- pexp_cutpoints
    class(fit) <- "survgbm"
    H_final <- survgbm_pexp_cumhaz(fit, xmat, max(pexp_cutpoints))[, 1]
    fit$fitted_raw <- log(pmax(H_final, 1e-300))
    fit$fitted <- H_final
  } else {
    fit$fitted_raw <- .Call("survgbm_predict_cpp", fit, xmat, "link")
    fit$fitted <- survgbm_transform_response(fit$fitted_raw, objective)
    if (objective == "cox") {
      fit$baseline <- survgbm_survival_baseline(time, status, fit$fitted_raw)
    } else {
      fit$survival_sigma <- 1
    }
  }
  class(fit) <- "survgbm"
  fit
}
