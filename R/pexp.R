## Piecewise-exponential (PEM) helpers: person-time expansion for fitting, and
## multi-interval evaluation for prediction. The gradient/Hessian this rests on
## (src/fastgbm.cpp: compute_pexp_grad_hess/compute_pexp_loss) is an ordinary
## Poisson-with-offset objective evaluated on the expanded data -- validated
## against finite differences and a Poisson-GLM equivalence check in
## tests/testthat/test-pexp.R.

fastgbm_pexp_cutpoints <- function(time, status, bins = 10L) {
  event_times <- time[status == 1L]
  if (length(unique(event_times)) < 2L) event_times <- time
  probs <- seq(0, 1, length.out = bins + 1L)
  cuts <- unique(as.numeric(stats::quantile(event_times, probs = probs, names = FALSE, type = 7)))
  if (length(cuts) < 2L) cuts <- c(0, max(time))
  cuts[1] <- 0
  cuts[length(cuts)] <- max(cuts[length(cuts)], max(time))
  cuts
}

# Expands (x, time, status) into person-time rows: one row per subject per
# interval they were at risk in, with `exposure` (time at risk in that
# interval) and `event` (1 only for the interval containing the subject's
# actual event, else 0). Appends log(interval midpoint) as an extra feature
# column, so the boosted ensemble models the hazard's shape over time jointly
# with the covariates rather than assuming a fixed post-hoc baseline.
fastgbm_expand_person_time <- function(x, time, status, cutpoints) {
  n <- nrow(x)
  nbins <- length(cutpoints) - 1L
  subject_list <- vector("list", n)
  for (i in seq_len(n)) {
    rows_i <- list()
    for (j in seq_len(nbins)) {
      c0 <- cutpoints[j]; c1 <- cutpoints[j + 1L]
      if (time[i] <= c0) break
      exposure <- min(time[i], c1) - c0
      event <- as.integer(status[i] == 1L && time[i] > c0 && time[i] <= c1)
      midpoint <- (c0 + min(c1, time[i])) / 2
      rows_i[[length(rows_i) + 1L]] <- c(subject = i, interval = j, midpoint = midpoint, exposure = exposure, event = event)
      if (time[i] <= c1) break
    }
    subject_list[[i]] <- do.call(rbind, rows_i)
  }
  meta <- do.call(rbind, subject_list)
  x_expanded <- cbind(x[meta[, "subject"], , drop = FALSE], log_time_mid = log(meta[, "midpoint"]))
  list(x = x_expanded, exposure = as.numeric(meta[, "exposure"]),
      event = as.integer(meta[, "event"]), subject = as.integer(meta[, "subject"]),
      interval = as.integer(meta[, "interval"]))
}

# Cumulative hazard H(t | x) for each row of `x` (original covariate
# representation, without the log-time column) at each requested time.
# Evaluates the ensemble once per (row, interval) pair up to each time (hazard
# is piecewise-constant, so H accrues linearly within an interval), and
# extrapolates past the model's last cutpoint by holding the final interval's
# hazard rate constant.
fastgbm_pexp_cumhaz <- function(object, x, times) {
  cutpoints <- object$pexp_cutpoints
  nbins <- length(cutpoints) - 1L
  midpoints <- (cutpoints[-length(cutpoints)] + cutpoints[-1]) / 2
  n <- nrow(x)

  x_rep <- x[rep(seq_len(n), times = nbins), , drop = FALSE]
  time_rep <- rep(log(midpoints), each = n)
  x_eval <- cbind(x_rep, log_time_mid = time_rep)
  pred <- .Call("fastgbm_predict_cpp", object, x_eval, "link")
  hazard <- matrix(exp(pred), nrow = n, ncol = nbins)

  widths <- diff(cutpoints)
  cumhaz_at_cut <- t(apply(hazard, 1, function(h) cumsum(h * widths)))
  cumhaz_at_cut <- cbind(0, cumhaz_at_cut)

  out <- matrix(NA_real_, nrow = n, ncol = length(times))
  for (k in seq_along(times)) {
    t <- times[k]
    j <- findInterval(t, cutpoints)
    j <- pmin(pmax(j, 1L), nbins)
    out[, k] <- cumhaz_at_cut[cbind(seq_len(n), j)] + (t - cutpoints[j]) * hazard[cbind(seq_len(n), j)]
  }
  out
}
