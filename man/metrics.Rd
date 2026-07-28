\name{metrics}
\alias{metrics}
\alias{rmse}
\alias{logloss}
\alias{accuracy}
\title{Evaluation helpers}
\description{Simple metrics for regression and binary classification.}
\usage{
metrics(object, newdata = NULL, y = NULL, type = c("response", "link"))
rmse(y, pred)
logloss(y, pred)
accuracy(y, pred)
}
\arguments{
\item{object}{A fitted fastgbm model.}
\item{newdata}{Optional new data.}
\item{y}{Optional observed outcomes.}
\item{type}{Prediction type.}
\item{pred}{Predictions.}
}
\value{Numeric metric values.}
