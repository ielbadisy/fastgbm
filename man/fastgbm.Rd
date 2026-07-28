\name{fastgbm}
\alias{fastgbm}
\alias{save_fastgbm}
\alias{load_fastgbm}
\alias{importance}
\alias{coef.fastgbm}
\title{Fast gradient boosting}
\description{Fit a histogram-based gradient boosting model with a compiled backend.}
\usage{
fastgbm(x, y = NULL, objective = NULL, ntrees = 500L, learning_rate = 0.05, max_depth = 6L,
  min_node_size = 20L, max_leaves = NULL, max_bins = 255L, subsample = 1, colsample = 1,
  lambda = 1, gamma = 0, min_child_weight = 1, validation = NULL, early_stopping = NULL,
  grow_policy = "depthwise", threads = 0L, seed = 1L, verbose = TRUE, ...)
save_fastgbm(object, path)
load_fastgbm(path)
importance(object, ...)
coef.fastgbm(object, ...)
}
\arguments{
\item{x}{Feature matrix or formula.}
\item{y}{Response vector or formula data.}
\item{objective}{Objective name.}
\item{ntrees}{Number of boosting rounds.}
\item{learning_rate}{Shrinkage parameter.}
\item{max_depth}{Maximum tree depth.}
\item{min_node_size}{Minimum rows in a node before splitting.}
\item{max_leaves}{Unused placeholder for compatibility.}
\item{max_bins}{Maximum number of histogram bins.}
\item{subsample}{Row subsampling fraction.}
\item{colsample}{Feature subsampling fraction.}
\item{lambda}{L2 regularization.}
\item{gamma}{Split penalty.}
\item{min_child_weight}{Minimum child Hessian sum.}
\item{validation}{Validation data placeholder.}
\item{early_stopping}{Early stopping placeholder.}
\item{grow_policy}{Tree growth policy.}
\item{threads}{Thread count placeholder.}
\item{seed}{Random seed.}
\item{verbose}{Whether to print progress.}
\item{object}{A fitted fastgbm object.}
\item{path}{Path to a serialized object.}
\item{...}{Additional arguments.}
}
\value{A fitted fastgbm object or a serialized model path.}
\examples{
\dontrun{
fit <- fastgbm(iris[, 1:4], as.numeric(iris$Species == "setosa"), objective = "binary:logistic")
predict(fit, iris[1:5, 1:4], type = "response")
}
}
