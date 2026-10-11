## Competing risks: the three CoxBoost options (Binder et al. 2009), with trees
## in place of componentwise linear updates.
##   "sh"   Fine-Gray subdistribution hazard for one cause, IPCW risk sets
##          (compiled objective "finegray", src/fastgbm.cpp).
##   "csh"  one independent cause-specific Cox booster per cause, other causes
##          treated as censored.
##   "ccsh" coupled cause-specific Cox boosters: one shared tree per round,
##          splits ranked by Fisher's combination of the per-cause score
##          p-values (src/fastgbm.cpp: build_tree_multi()).
## Cumulative incidence predictions follow CoxBoost's predict(type = "CIF").

#' @noRd
fastgbm_is_comrisk <- function(time, status, y, comrisk) {
  if (!is.null(comrisk)) {
    return(TRUE)
  }
  if (inherits(y, "Surv")) {
    return(identical(attr(y, "type"), "mright"))
  }
  !is.null(status) && is.numeric(status) && any(status == 0) &&
    length(unique(status[!is.na(status)])) >= 3L
}

# Returns integer cause codes (0 = censored, 1..K) and the cause labels.
#' @noRd
fastgbm_comrisk_response <- function(time, status = NULL, y = NULL, causes = NULL) {
  labels <- NULL
  if (inherits(y, "Surv")) {
    time <- y[, "time"]
    status <- y[, "status"]
    if (identical(attr(y, "type"), "mright")) {
      labels <- attr(y, "states")
    }
  }
  if (is.null(time) || is.null(status)) {
    stop("`time` and `status` are required, or a `survival::Surv` response.", call. = FALSE)
  }
  time <- as.numeric(time)
  if (length(time) != length(status)) {
    stop("`time` and `status` must have the same length.", call. = FALSE)
  }
  if (any(!is.finite(time)) || any(time <= 0)) {
    stop("Survival times must be positive and finite.", call. = FALSE)
  }
  if (anyNA(status)) {
    stop("`status` must not contain missing values.", call. = FALSE)
  }
  if (is.null(labels)) {
    codes <- sort(unique(status[status != 0]))
    labels <- as.character(codes)
    status <- match(status, codes, nomatch = 0L)
  }
  status <- as.integer(status)
  if (!is.null(causes)) {
    # Validation data: map onto the training causes.
    lab <- ifelse(status == 0L, NA_character_, labels[pmax(status, 1L)])
    if (any(!is.na(lab) & !lab %in% causes)) {
      stop("`validation` contains causes not present in the training data.", call. = FALSE)
    }
    status <- match(lab, causes, nomatch = 0L)
    labels <- causes
  }
  list(time = time, status = status, causes = labels)
}

# Kaplan-Meier estimate of the censoring distribution G, as a right-continuous
# step function (CoxBoost: survfit(Surv(time, status == 0) ~ 1)).
#' @noRd
fastgbm_censoring_km <- function(time, status) {
  ut <- sort(unique(time))
  at_risk <- length(time) - match(ut, sort(time)) + 1
  d_cens <- tabulate(match(time[status == 0L], ut), nbins = length(ut))
  list(times = ut, surv = cumprod(1 - d_cens / at_risk))
}

#' @noRd
fastgbm_eval_step <- function(step, t, at_zero = 1) {
  if (!length(step$times)) {
    return(rep(at_zero, length(t)))
  }
  approx(c(0, step$times), c(at_zero, step$surv), xout = t, method = "constant",
         rule = 2, f = 0)$y
}

# Breslow baseline of the Fine-Gray model, using the IPCW-weighted risk sets of
# the compiled "finegray" objective (CoxBoost calc.Lambda()):
# Lambda0(t) = sum_{u <= t} d_u / R(u),
# R(u) = sum_{T_j >= u} e_j + G(u) * sum_{competing j, T_j < u} e_j / G(T_j).
# `status` is coded 1 = cause of interest, 2 = competing, 0 = censored.
#' @noRd
fastgbm_finegray_baseline <- function(time, status, ipcw, lp) {
  ev <- sort(unique(time[status == 1L]))
  if (!length(ev)) {
    return(list(times = numeric(), cumhaz = numeric()))
  }
  e <- exp(pmin(lp, 35))
  o <- order(time)
  ts <- time[o]
  at_risk <- rev(cumsum(rev(e[o])))[findInterval(ev, ts, left.open = TRUE) + 1L]
  comp <- status == 2L
  tc <- time[comp]
  oc <- order(tc)
  csum <- c(0, cumsum((e[comp] / pmax(ipcw[comp], 1e-12))[oc]))
  competing <- csum[findInterval(ev, tc[oc], left.open = TRUE) + 1L]
  g_ev <- ipcw[match(ev, time)]
  d <- tabulate(match(time[status == 1L], ev), nbins = length(ev))
  list(times = ev, cumhaz = cumsum(d / pmax(at_risk + g_ev * competing, 1e-12)))
}

#' @noRd
fastgbm_comrisk_fit <- function(xmat, time, status, y, comrisk, cause, ntrees,
                                learning_rate, max_depth, min_node_size, max_bins,
                                subsample, colsample, lambda, gamma, min_child_weight,
                                validation, early_stopping, threads, seed, verbose,
                                call) {
  comrisk <- match.arg(comrisk, c("sh", "csh", "ccsh"))
  resp <- fastgbm_comrisk_response(time, status, y)
  time <- resp$time
  status <- resp$status
  causes <- resp$causes
  if (length(time) != nrow(xmat)) {
    stop("`time`/`status` must have the same length as `nrow(x)`.", call. = FALSE)
  }
  K <- length(causes)
  if (K < 2L) {
    stop("Competing risks need at least two causes in `status` (0 = censored).", call. = FALSE)
  }
  if (xor(is.null(validation), is.null(early_stopping))) {
    stop("`validation` and `early_stopping` must be supplied together.", call. = FALSE)
  }
  resp_valid <- NULL
  if (!is.null(validation)) {
    if (is.null(validation$x)) {
      stop("`validation` must include `x`, in the same matrix representation as the training data.", call. = FALSE)
    }
    resp_valid <- fastgbm_comrisk_response(validation$time, validation$status, validation$y, causes)
  }
  hyper <- list(
    ntrees = ntrees, learning_rate = learning_rate, max_depth = max_depth,
    min_node_size = min_node_size, max_bins = max_bins, subsample = subsample,
    colsample = colsample, lambda = lambda, gamma = gamma,
    min_child_weight = min_child_weight, threads = threads
  )

  censoring <- NULL
  if (comrisk == "sh") {
    if (is.null(cause)) {
      cause <- causes[1L]
    }
    cause <- as.character(cause)
    if (length(cause) != 1L || !cause %in% causes) {
      stop("`cause` must be one of the causes in `status`: ", paste(causes, collapse = ", "), ".", call. = FALSE)
    }
    k <- match(cause, causes)
    status_fg <- ifelse(status == 0L, 0L, ifelse(status == k, 1L, 2L))
    censoring <- fastgbm_censoring_km(time, status)
    ipcw <- fastgbm_eval_step(censoring, time)
    fits <- list(fastgbm_finegray_fit(
      xmat, time, status_fg, ipcw, hyper, validation, resp_valid, k, censoring,
      early_stopping, seed, verbose
    ))
    names(fits) <- cause
  } else if (comrisk == "csh") {
    fits <- lapply(seq_len(K), function(k) {
      validation_k <- if (is.null(validation)) NULL else
        list(x = validation$x, time = resp_valid$time, status = as.integer(resp_valid$status == k))
      fit_k <- do.call(fastgbm, c(list(
        xmat, time = time, status = as.integer(status == k), objective = "cox",
        validation = validation_k, early_stopping = if (is.null(validation_k)) NULL else early_stopping,
        seed = seed + k - 1L, verbose = FALSE
      ), hyper))
      fit_k$call <- NULL
      fit_k
    })
    names(fits) <- causes
  } else {
    fits <- fastgbm_ccsh_fit(xmat, time, status, K, hyper, validation, resp_valid,
                             early_stopping, seed, verbose)
    names(fits) <- causes
  }

  structure(
    list(
      call = call,
      objective = "cox",
      comrisk = comrisk,
      causes = if (comrisk == "sh") cause else causes,
      all_causes = causes,
      fits = fits,
      censoring = censoring,
      surv_time = time,
      surv_status = status,
      ntrees = vapply(fits, function(f) f$ntrees, integer(1)),
      learning_rate = learning_rate,
      max_depth = max_depth,
      fitted_raw = vapply(fits, function(f) f$fitted_raw, numeric(length(time)))
    ),
    class = "fastgbm_comrisk"
  )
}

#' @noRd
fastgbm_finegray_fit <- function(xmat, time, status_fg, ipcw, hyper, validation,
                                 resp_valid, k, censoring, early_stopping, seed, verbose) {
  has_validation <- !is.null(validation)
  x_valid <- matrix(numeric(0), nrow = 0, ncol = ncol(xmat))
  time_valid <- numeric(0)
  status_valid <- integer(0)
  ipcw_valid <- numeric(0)
  if (has_validation) {
    x_valid <- fastgbm_as_matrix(validation$x)
    if (ncol(x_valid) != ncol(xmat) || nrow(x_valid) != length(resp_valid$time)) {
      stop("`validation$x` must match the training columns and the validation response rows.", call. = FALSE)
    }
    time_valid <- resp_valid$time
    sv <- resp_valid$status
    status_valid <- ifelse(sv == 0L, 0L, ifelse(sv == k, 1L, 2L))
    ipcw_valid <- fastgbm_eval_step(censoring, time_valid)
  }
  fit <- .Call(
    "fastgbm_fit_cpp", xmat, as.numeric(time), as.integer(status_fg), "finegray",
    as.integer(hyper$ntrees), as.numeric(hyper$learning_rate), as.integer(hyper$max_depth),
    as.integer(hyper$min_node_size), as.integer(hyper$max_bins), as.numeric(hyper$subsample),
    as.numeric(hyper$colsample), as.numeric(hyper$lambda), as.numeric(hyper$gamma),
    as.numeric(hyper$min_child_weight), as.integer(seed), as.integer(hyper$threads),
    as.logical(verbose), x_valid, as.numeric(time_valid), as.integer(status_valid),
    if (has_validation) as.integer(early_stopping) else 0L,
    as.numeric(ipcw), as.numeric(ipcw_valid)
  )
  fit$objective <- "finegray"
  fit$learning_rate <- hyper$learning_rate
  fit$max_depth <- hyper$max_depth
  fit$min_node_size <- hyper$min_node_size
  fit$n_trees_grown <- length(fit$trees)
  if (has_validation) {
    fit$trees <- fit$trees[seq_len(fit$best_iteration)]
  }
  fit$ntrees <- length(fit$trees)
  fit$fitted_raw <- .Call("fastgbm_predict_cpp", fit, xmat, "link")
  fit$baseline <- fastgbm_finegray_baseline(time, status_fg, ipcw, fit$fitted_raw)
  class(fit) <- "fastgbm"
  fit
}

#' @noRd
fastgbm_ccsh_fit <- function(xmat, time, status, K, hyper, validation, resp_valid,
                             early_stopping, seed, verbose) {
  has_validation <- !is.null(validation)
  x_valid <- matrix(numeric(0), nrow = 0, ncol = ncol(xmat))
  time_valid <- numeric(0)
  status_valid <- integer(0)
  if (has_validation) {
    x_valid <- fastgbm_as_matrix(validation$x)
    if (ncol(x_valid) != ncol(xmat) || nrow(x_valid) != length(resp_valid$time)) {
      stop("`validation$x` must match the training columns and the validation response rows.", call. = FALSE)
    }
    time_valid <- resp_valid$time
    status_valid <- resp_valid$status
  }
  res <- .Call(
    "fastgbm_fit_ccsh_cpp", xmat, as.numeric(time), as.integer(status), as.integer(K),
    as.integer(hyper$ntrees), as.numeric(hyper$learning_rate), as.integer(hyper$max_depth),
    as.integer(hyper$min_node_size), as.integer(hyper$max_bins), as.numeric(hyper$subsample),
    as.numeric(hyper$colsample), as.numeric(hyper$lambda), as.numeric(hyper$gamma),
    as.numeric(hyper$min_child_weight), as.integer(seed), as.integer(hyper$threads),
    as.logical(verbose), x_valid, as.numeric(time_valid), as.integer(status_valid),
    if (has_validation) as.integer(early_stopping) else 0L
  )
  lapply(seq_len(K), function(k) {
    fit <- res$fits[[k]]
    fit$n_trees_grown <- length(fit$trees)
    fit$trees <- fit$trees[seq_len(res$best_iteration)]
    fit$ntrees <- length(fit$trees)
    fit$best_iteration <- res$best_iteration
    fit$validation_history <- res$validation_history
    fit$stopping_reason <- res$stopping_reason
    fit$fitted_raw <- .Call("fastgbm_predict_cpp", fit, xmat, "link")
    fit$baseline <- fastgbm_survival_baseline(time, as.integer(status == k), fit$fitted_raw)
    class(fit) <- "fastgbm"
    fit
  })
}

# Cause-specific cumulative incidence, as in CoxBoost's predict(type = "CIF"):
# the all-cause survival exp(-sum_k H_k) on the grid of all event times, each
# cause's hazard spread uniformly between its own event times, integrated
# against the interval-midpoint all-cause survival, then rescaled so that the
# incidences add up to 1 - S(t).
#' @noRd
fastgbm_cif_cause_specific <- function(lp, baselines, times) {
  n <- nrow(lp)
  K <- ncol(lp)
  grid <- sort(unique(unlist(lapply(baselines, `[[`, "times"))))
  if (!length(grid) || grid[1] > 0) grid <- c(0, grid)
  if (max(grid) < max(times)) {
    grid <- c(grid, max(times))
  } else {
    grid <- grid[grid <= max(times)]
  }
  if (length(grid) < 2L) {
    return(rep(list(matrix(0, n, length(times))), K))
  }
  cumhaz_at <- function(k, t) {
    outer(exp(pmin(lp[, k], 35)), fastgbm_eval_step(list(times = baselines[[k]]$times, surv = baselines[[k]]$cumhaz), t, at_zero = 0))
  }
  all_surv <- exp(-Reduce(`+`, lapply(seq_len(K), cumhaz_at, t = grid)))
  mid_surv <- (all_surv[, -ncol(all_surv), drop = FALSE] + all_surv[, -1, drop = FALSE]) / 2
  dgrid <- diff(grid)
  cif <- lapply(seq_len(K), function(k) {
    et <- baselines[[k]]$times
    if (!length(et)) return(matrix(0, n, length(grid)))
    if (et[1] != 0) et <- c(0, et)
    H <- cumhaz_at(k, et)
    haz <- cbind((H[, -1, drop = FALSE] - H[, -ncol(H), drop = FALSE]) /
                   rep(diff(et), each = n), 0)
    idx <- findInterval(grid[-length(grid)], et)
    inc <- haz[, idx, drop = FALSE] * rep(dgrid, each = n) * mid_surv
    cbind(0, t(apply(inc, 1, cumsum)))
  })
  total <- Reduce(`+`, cif)
  scale <- ifelse(total > 0, (1 - all_surv) / total, 0)
  scale[, 1] <- 0
  at <- findInterval(times, grid)
  lapply(cif, function(m) (m * scale)[, at, drop = FALSE])
}

#' Predict from a fitted competing-risks fastgbm model
#'
#' @param object A fitted `fastgbm_comrisk` object (from [fastgbm()] with
#'   `comrisk`).
#' @param newdata New data to predict on, in the same representation used to fit.
#' @param type `"cif"` (cumulative incidence at `times`), `"link"` (the
#'   boosted log subdistribution hazard ratio for `"sh"`, or the log
#'   cause-specific hazard ratios for `"csh"`/`"ccsh"`), or `"response"`
#'   (exp of `"link"`).
#' @param times Required when `type = "cif"`: times at which to evaluate the
#'   cumulative incidence.
#' @param ... Unused.
#' @return For `"cif"`, a named list with one `nrow(newdata)` by
#'   `length(times)` matrix per modelled cause (a single cause for `"sh"`).
#'   Otherwise a matrix with one column per modelled cause.
#' @export
predict.fastgbm_comrisk <- function(object, newdata, type = c("cif", "link", "response"),
                                    times = NULL, ...) {
  type <- match.arg(type)
  if (!is.null(object$formula_terms) && is.data.frame(newdata)) {
    mf <- model.frame(object$formula_terms, data = newdata, na.action = na.pass, xlev = object$xlevels)
    newdata <- model.matrix(delete.response(object$formula_terms), data = mf, contrasts.arg = object$contrasts)
  } else {
    newdata <- fastgbm_as_matrix(newdata)
  }
  lp <- vapply(object$fits, function(f) .Call("fastgbm_predict_cpp", f, newdata, "link"),
               numeric(nrow(newdata)))
  lp <- matrix(lp, nrow = nrow(newdata), dimnames = list(NULL, object$causes))
  if (type == "link") return(lp)
  if (type == "response") return(exp(lp))
  if (is.null(times)) {
    stop("`times` is required for `type = \"cif\"`.", call. = FALSE)
  }
  if (object$comrisk == "sh") {
    b <- object$fits[[1]]$baseline
    H0 <- fastgbm_eval_step(list(times = b$times, surv = b$cumhaz), times, at_zero = 0)
    out <- list(1 - exp(-outer(exp(pmin(lp[, 1], 35)), H0)))
  } else {
    out <- fastgbm_cif_cause_specific(lp, lapply(object$fits, `[[`, "baseline"), times)
  }
  names(out) <- object$causes
  out
}

#' Print a fitted competing-risks fastgbm model
#'
#' @param x A fitted `fastgbm_comrisk` object.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.fastgbm_comrisk <- function(x, ...) {
  label <- c(sh = "Fine-Gray subdistribution hazard",
             csh = "cause-specific hazards",
             ccsh = "coupled cause-specific hazards")[[x$comrisk]]
  cat("fastgbm competing risks model (", label, ")\n", sep = "")
  cat("  causes:", paste(x$all_causes, collapse = ", "), "\n")
  if (x$comrisk == "sh") cat("  modelled cause:", x$causes, "\n")
  cat("  trees per cause:", paste(x$ntrees, collapse = ", "), "\n")
  cat("  learning rate:", x$learning_rate, "\n")
  cat("  max depth:", x$max_depth, "\n")
  invisible(x)
}
