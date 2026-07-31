library(fastgbm)
library(survival)
library(gbm)
library(xgboost)
library(ranger)

set.seed(42)
dat <- na.omit(as.data.frame(lung[, c("time", "status", "age", "sex", "ph.ecog", "ph.karno", "pat.karno", "meal.cal", "wt.loss")]))
dat$status01 <- as.integer(dat$status == max(dat$status))
idx <- sample.int(nrow(dat))
train_idx <- idx[seq_len(floor(0.7 * length(idx)))]
test_idx <- idx[-seq_len(floor(0.7 * length(idx)))]
train <- dat[train_idx, , drop = FALSE]
test <- dat[test_idx, , drop = FALSE]

form <- Surv(time, status01) ~ age + sex + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss
x_train <- model.matrix(delete.response(terms(form)), data = train)
x_test <- model.matrix(delete.response(terms(form)), data = test)

score_cindex <- function(time, status, risk) {
  df <- data.frame(time = time, status = status, risk = risk)
  survival::concordance(Surv(time, status) ~ risk, data = df)$concordance
}

timed <- function(expr) {
  gc()
  t <- system.time(value <- force(expr))
  list(value = value, elapsed = unname(t["elapsed"]))
}

results <- list()

fit_fastgbm <- timed(
  fastgbm(
    x_train,
    time = train$time,
    status = train$status01,
    objective = "cox",
    ntrees = 100L,
    learning_rate = 0.05,
    max_depth = 3L,
    min_node_size = 10L,
    seed = 42L,
    verbose = FALSE
  )
)
pred_fastgbm <- timed(predict(fit_fastgbm$value, x_test, type = "link"))
results[[length(results) + 1L]] <- data.frame(
  model = "fastgbm",
  train_sec = fit_fastgbm$elapsed,
  predict_sec = pred_fastgbm$elapsed,
  cindex = score_cindex(test$time, test$status01, pred_fastgbm$value),
  stringsAsFactors = FALSE
)

fit_gbm <- timed(
  gbm(
    form,
    data = train,
    distribution = "coxph",
    n.trees = 100L,
    interaction.depth = 3L,
    shrinkage = 0.05,
    n.minobsinnode = 10L,
    bag.fraction = 1,
    train.fraction = 1,
    verbose = FALSE
  )
)
pred_gbm <- timed(predict(fit_gbm$value, newdata = test, n.trees = 100L, type = "link"))
results[[length(results) + 1L]] <- data.frame(
  model = "gbm",
  train_sec = fit_gbm$elapsed,
  predict_sec = pred_gbm$elapsed,
  cindex = score_cindex(test$time, test$status01, pred_gbm$value),
  stringsAsFactors = FALSE
)

label_xgb <- ifelse(train$status01 == 1, train$time, -train$time)
fit_xgb <- timed(
  xgb.train(
    params = list(
      objective = "survival:cox",
      eta = 0.05,
      max_depth = 3L,
      min_child_weight = 1,
      subsample = 1,
      colsample_bytree = 1,
      verbosity = 0
    ),
    data = xgb.DMatrix(x_train, label = label_xgb),
    nrounds = 100L,
    verbose = 0
  )
)
pred_xgb <- timed(predict(fit_xgb$value, xgb.DMatrix(x_test)))
results[[length(results) + 1L]] <- data.frame(
  model = "xgboost",
  train_sec = fit_xgb$elapsed,
  predict_sec = pred_xgb$elapsed,
  cindex = score_cindex(test$time, test$status01, pred_xgb$value),
  stringsAsFactors = FALSE
)

fit_ranger <- timed(
  ranger(
    form,
    data = train,
    num.trees = 100L,
    mtry = max(1L, floor(sqrt(ncol(x_train)))),
    min.node.size = 10L,
    splitrule = "logrank",
    seed = 42L,
    write.forest = TRUE,
    importance = "none"
  )
)
ranger_pred <- timed(predict(fit_ranger$value, data = test))
ranger_risk <- rowSums(ranger_pred$value$chf)
results[[length(results) + 1L]] <- data.frame(
  model = "ranger",
  train_sec = fit_ranger$elapsed,
  predict_sec = ranger_pred$elapsed,
  cindex = score_cindex(test$time, test$status01, ranger_risk),
  stringsAsFactors = FALSE
)

results <- do.call(rbind, results)
print(results)
write.csv(results, file = "inst/benchmarks/survival-cox-lung.csv", row.names = FALSE)
