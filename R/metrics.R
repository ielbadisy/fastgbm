rmse <- function(y, pred) {
  sqrt(mean((as.numeric(y) - as.numeric(pred))^2))
}

logloss <- function(y, pred) {
  y <- as.numeric(y)
  p <- pmin(pmax(as.numeric(pred), 1e-15), 1 - 1e-15)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

accuracy <- function(y, pred) {
  mean((as.numeric(y) > 0.5) == (as.numeric(pred) > 0.5))
}
