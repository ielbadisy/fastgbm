#include <Rcpp.h>
#include <RcppParallel.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

using namespace Rcpp;
using namespace RcppParallel;

struct TreeNode {
  int feature = -1;
  int threshold = -1;
  bool missing_left = true;
  int left = -1;
  int right = -1;
  double value = 0.0;
};

struct TreeModel {
  std::vector<TreeNode> nodes;
};

struct NodeSplit {
  int feature = -1;
  int threshold = -1;
  bool missing_left = true;
  double gain = -INFINITY;
};

static std::vector<double> make_cuts(const std::vector<double>& values, int max_bins) {
  std::vector<double> v;
  v.reserve(values.size());
  for (double x : values) {
    if (R_finite(x)) v.push_back(x);
  }
  if (v.empty()) return {};
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  if (static_cast<int>(v.size()) <= 1) return {};
  int bins = std::min(max_bins, static_cast<int>(v.size()));
  int cut_count = std::max(0, bins - 1);
  std::vector<double> cuts;
  cuts.reserve(cut_count);
  for (int j = 1; j <= cut_count; ++j) {
    double pos = (static_cast<double>(j) * (v.size() - 1)) / bins;
    size_t idx = static_cast<size_t>(std::floor(pos));
    if (idx >= v.size()) idx = v.size() - 1;
    cuts.push_back(v[idx]);
  }
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  return cuts;
}

static int bin_value(double x, const std::vector<double>& cuts) {
  if (!R_finite(x)) return 0;
  auto it = std::upper_bound(cuts.begin(), cuts.end(), x);
  return static_cast<int>(it - cuts.begin()) + 1;
}

static List make_tree_list(const TreeModel& tree) {
  int n = static_cast<int>(tree.nodes.size());
  IntegerVector feature(n);
  IntegerVector threshold(n);
  LogicalVector missing_left(n);
  IntegerVector left(n);
  IntegerVector right(n);
  NumericVector value(n);
  for (int i = 0; i < n; ++i) {
    const TreeNode& node = tree.nodes[i];
    feature[i] = node.feature;
    threshold[i] = node.threshold;
    missing_left[i] = node.missing_left;
    left[i] = node.left;
    right[i] = node.right;
    value[i] = node.value;
  }
  return List::create(
    _["feature"] = feature,
    _["threshold"] = threshold,
    _["missing_left"] = missing_left,
    _["left"] = left,
    _["right"] = right,
    _["value"] = value
  );
}

// Computes the best split for a subset of features (indices [begin, end) into
// `feature_ids`) into `results`, one entry per feature. Reads Rcpp objects only
// through the thread-safe RMatrix/RVector wrappers, per RcppParallel's contract
// that raw Rcpp/R objects must not be touched from worker threads. Writes are to
// disjoint `results[idx]` slots, so calling this directly (serial) or through
// `parallelFor` (parallel) produces bit-identical results regardless of thread
// count or scheduling -- the split search is a data-parallel reduction over
// features, not an accumulation, so there is no order-dependent floating-point
// summation across threads.
struct SplitFinder : public Worker {
  const RMatrix<int> bins;
  const RVector<double> grad;
  const RVector<double> hess;
  const std::vector<int>& node_rows;
  const std::vector<int>& feature_ids;
  double G, H, lambda, gamma, min_child_weight;
  std::vector<NodeSplit>& results;

  SplitFinder(const IntegerMatrix& bins_, const NumericVector& grad_, const NumericVector& hess_,
             const std::vector<int>& node_rows_, const std::vector<int>& feature_ids_,
             double G_, double H_, double lambda_, double gamma_, double min_child_weight_,
             std::vector<NodeSplit>& results_)
    : bins(bins_), grad(grad_), hess(hess_), node_rows(node_rows_), feature_ids(feature_ids_),
      G(G_), H(H_), lambda(lambda_), gamma(gamma_), min_child_weight(min_child_weight_),
      results(results_) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t idx = begin; idx < end; ++idx) {
      int feature = feature_ids[idx];
      int max_bin = 0;
      for (int r : node_rows) {
        int b = bins(r, feature);
        if (b > max_bin) max_bin = b;
      }
      NodeSplit local;
      if (max_bin < 2) {
        results[idx] = local;
        continue;
      }

      std::vector<double> gsum(max_bin + 1, 0.0), hsum(max_bin + 1, 0.0);
      for (int r : node_rows) {
        int b = bins(r, feature);
        if (b < 0) b = 0;
        if (b > max_bin) b = max_bin;
        gsum[b] += grad[r];
        hsum[b] += hess[r];
      }

      double missing_g = gsum[0];
      double missing_h = hsum[0];
      double left_g = 0.0, left_h = 0.0;
      for (int t = 1; t < max_bin; ++t) {
        left_g += gsum[t];
        left_h += hsum[t];
        double right_g = G - left_g - missing_g;
        double right_h = H - left_h - missing_h;
        if (left_h < min_child_weight || right_h < min_child_weight) continue;

        for (int missing_side = 0; missing_side < 2; ++missing_side) {
          double GL = left_g + (missing_side == 0 ? missing_g : 0.0);
          double HL = left_h + (missing_side == 0 ? missing_h : 0.0);
          double GR = right_g + (missing_side == 1 ? missing_g : 0.0);
          double HR = right_h + (missing_side == 1 ? missing_h : 0.0);
          if (HL < min_child_weight || HR < min_child_weight) continue;
          double gain = 0.5 * (
            (GL * GL) / (HL + lambda) +
            (GR * GR) / (HR + lambda) -
            ((GL + GR) * (GL + GR)) / (HL + HR + lambda)
          ) - gamma;
          if (gain > local.gain) {
            local.feature = feature;
            local.threshold = t;
            local.missing_left = (missing_side == 0);
            local.gain = gain;
          }
        }
      }
      results[idx] = local;
    }
  }
};

// Feature subsampling happens per NODE (like ranger's `mtry`, resampled at
// every split), not once per tree (xgboost's `colsample_bytree`): a fresh
// random subset is drawn for every node's split search, via the same
// Bernoulli(colsample) scheme used previously at the tree level. This is a
// well-established technique (equivalent to xgboost's `colsample_bynode`),
// not a novel invention -- it decorrelates sibling/descendant splits within a
// single tree far more than per-tree sampling does, which is the core
// mechanism behind random forests' variance reduction over boosting's
// sequential, correlated trees. `rng` is advanced in strict node-visit order
// (depth-first, left child before right), which is itself independent of how
// many threads the split search inside a node uses, so this preserves the
// existing determinism-across-thread-counts guarantee.
static TreeModel build_tree(const IntegerMatrix& bins, const NumericVector& grad,
                            const NumericVector& hess, const std::vector<int>& rows,
                            int max_depth, int min_node_size, double lambda,
                            double gamma, double min_child_weight,
                            int p, double colsample, std::mt19937& rng,
                            std::vector<double>& importance,
                            int num_threads) {
  TreeModel tree;
  tree.nodes.reserve(2 * rows.size() + 1);
  std::uniform_real_distribution<double> ur(0.0, 1.0);

  std::function<int(const std::vector<int>&, int)> grow = [&](const std::vector<int>& node_rows, int depth) -> int {
    int node_id = static_cast<int>(tree.nodes.size());
    tree.nodes.push_back(TreeNode{});

    double G = 0.0, H = 0.0;
    for (int r : node_rows) {
      G += grad[r];
      H += hess[r];
    }
    double leaf_value = -G / (H + lambda);
    tree.nodes[node_id].value = leaf_value;

    if (depth >= max_depth || static_cast<int>(node_rows.size()) < 2 * min_node_size || H < min_child_weight) {
      return node_id;
    }

    std::vector<int> feature_ids;
    feature_ids.reserve(p);
    for (int j = 0; j < p; ++j) {
      if (colsample >= 1.0 || ur(rng) < colsample) feature_ids.push_back(j);
    }
    if (feature_ids.empty()) {
      for (int j = 0; j < p; ++j) feature_ids.push_back(j);
    }

    std::vector<NodeSplit> results(feature_ids.size());
    SplitFinder worker(bins, grad, hess, node_rows, feature_ids, G, H, lambda, gamma, min_child_weight, results);
    // Parallel dispatch overhead is not worth it for small nodes/feature counts;
    // below the threshold, call the worker directly (serial) for identical results.
    if (feature_ids.size() >= 8 && node_rows.size() >= 256) {
      parallelFor(0, feature_ids.size(), worker, /*grainSize=*/1, num_threads);
    } else {
      worker(0, feature_ids.size());
    }

    NodeSplit best;
    for (const NodeSplit& r : results) {
      if (r.feature >= 0 && r.gain > best.gain) best = r;
    }

    if (best.feature < 0 || !R_finite(best.gain) || best.gain <= 0.0) {
      return node_id;
    }

    std::vector<int> left_rows;
    std::vector<int> right_rows;
    left_rows.reserve(node_rows.size());
    right_rows.reserve(node_rows.size());
    for (int r : node_rows) {
      int b = bins(r, best.feature);
      if (b == 0) {
        if (best.missing_left) left_rows.push_back(r);
        else right_rows.push_back(r);
      } else if (b <= best.threshold) {
        left_rows.push_back(r);
      } else {
        right_rows.push_back(r);
      }
    }
    if (left_rows.empty() || right_rows.empty()) {
      return node_id;
    }

    tree.nodes[node_id].feature = best.feature;
    tree.nodes[node_id].threshold = best.threshold;
    tree.nodes[node_id].missing_left = best.missing_left;
    tree.nodes[node_id].left = grow(left_rows, depth + 1);
    tree.nodes[node_id].right = grow(right_rows, depth + 1);
    importance[best.feature] += std::max(0.0, best.gain);
    return node_id;
  };

  grow(rows, 0);
  return tree;
}

static double predict_tree_row(const TreeModel& tree, const IntegerMatrix& bins, int row) {
  int node = 0;
  while (true) {
    const TreeNode& n = tree.nodes[node];
    if (n.feature < 0) return n.value;
    int b = bins(row, n.feature);
    if (b == 0) {
      node = n.missing_left ? n.left : n.right;
    } else if (b <= n.threshold) {
      node = n.left;
    } else {
      node = n.right;
    }
  }
}

static inline double safe_exp(double x) {
  return std::exp(std::max(-35.0, std::min(35.0, x)));
}

static void compute_cox_grad_hess(const NumericVector& pred, const NumericVector& time,
                                  const IntegerVector& status, NumericVector& grad,
                                  NumericVector& hess) {
  int n = pred.size();
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (time[a] == time[b]) return a < b;
    return time[a] < time[b];
  });

  std::vector<double> exp_pred(n);
  for (int i = 0; i < n; ++i) exp_pred[i] = safe_exp(pred[i]);

  std::vector<double> uniq_times;
  std::vector<std::vector<int>> blocks;
  for (int idx = 0; idx < n; ) {
    double t = time[order[idx]];
    std::vector<int> block;
    while (idx < n && time[order[idx]] == t) {
      block.push_back(order[idx]);
      ++idx;
    }
    uniq_times.push_back(t);
    blocks.push_back(std::move(block));
  }

  std::vector<double> risk_sum(blocks.size(), 0.0);
  double running = 0.0;
  for (int b = static_cast<int>(blocks.size()) - 1; b >= 0; --b) {
    for (int idx : blocks[b]) {
      running += exp_pred[idx];
    }
    risk_sum[b] = running;
  }

  std::vector<double> cum1(blocks.size(), 0.0), cum2(blocks.size(), 0.0);
  double c1 = 0.0, c2 = 0.0;
  for (size_t b = 0; b < blocks.size(); ++b) {
    int d = 0;
    for (int idx : blocks[b]) {
      if (status[idx] == 1) ++d;
    }
    double denom = std::max(risk_sum[b], 1e-12);
    c1 += static_cast<double>(d) / denom;
    c2 += static_cast<double>(d) / (denom * denom);
    cum1[b] = c1;
    cum2[b] = c2;
  }

  for (size_t b = 0; b < blocks.size(); ++b) {
    for (int idx : blocks[b]) {
      double a = cum1[b];
      double b2 = cum2[b];
      double e = exp_pred[idx];
      grad[idx] = e * a - status[idx];
      hess[idx] = std::max(1e-6, e * a - e * e * b2);
    }
  }
}

static void compute_aft_grad_hess(const NumericVector& pred, const NumericVector& time,
                                  const IntegerVector& status, NumericVector& grad,
                                  NumericVector& hess, double sigma = 1.0) {
  int n = pred.size();
  double sigma2 = sigma * sigma;
  for (int i = 0; i < n; ++i) {
    double logt = std::log(time[i]);
    if (status[i] == 1) {
      grad[i] = (pred[i] - logt) / sigma2;
      hess[i] = 1.0 / sigma2;
    } else {
      // Censored: loss = -log S(z), z = (log t - pred) / sigma, S = survival function of Z.
      // d(loss)/d(pred) = -pdf(z) / (sigma * S(z));
      // d^2(loss)/d(pred)^2 = pdf(z) * (pdf(z) - z * S(z)) / (sigma^2 * S(z)^2).
      double z = (logt - pred[i]) / sigma;
      double sf = std::max(1e-12, R::pnorm5(z, 0.0, 1.0, /*lower_tail=*/false, /*log_p=*/false));
      double pdf = R::dnorm4(z, 0.0, 1.0, /*give_log=*/false);
      grad[i] = -pdf / (sigma * sf);
      double h = pdf * (pdf - z * sf) / (sigma2 * sf * sf);
      hess[i] = std::max(1e-6, h);
    }
  }
}

// Piecewise-exponential (PEM) objective, via the standard "Poisson trick": on
// person-time-expanded data (one row per subject-interval; see
// R/pexp.R:survgbm_expand_person_time()), `time` holds each row's exposure
// (time at risk within that interval) and `status` holds its event indicator
// (1 if the subject's actual event fell in that interval, else 0). The model
// predicts the log hazard rate for the row; the Poisson mean is
// exposure * exp(pred). This is an ordinary Poisson-with-offset gradient/
// Hessian, not an approximation -- unlike a Royston-Parmar-style joint hazard
// surface, no derivative of the ensemble w.r.t. the time feature is needed,
// because the hazard is directly exp(pred) rather than a spline whose local
// slope has to be estimated.
static void compute_pexp_grad_hess(const NumericVector& pred, const NumericVector& exposure,
                                   const IntegerVector& event, NumericVector& grad,
                                   NumericVector& hess) {
  int n = pred.size();
  for (int i = 0; i < n; ++i) {
    double mean_i = exposure[i] * safe_exp(pred[i]);
    grad[i] = mean_i - event[i];
    hess[i] = std::max(1e-6, mean_i);
  }
}

static double compute_pexp_loss(const NumericVector& pred, const NumericVector& exposure,
                                const IntegerVector& event) {
  int n = pred.size();
  double loss = 0.0;
  for (int i = 0; i < n; ++i) {
    loss += exposure[i] * safe_exp(pred[i]) - event[i] * pred[i];
  }
  return loss;
}

// Negative Cox partial log-likelihood (Breslow), reusing the same sorted-blocks/
// risk-sum machinery as compute_cox_grad_hess. Used as the early-stopping metric.
static double compute_cox_loss(const NumericVector& pred, const NumericVector& time,
                               const IntegerVector& status) {
  int n = pred.size();
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (time[a] == time[b]) return a < b;
    return time[a] < time[b];
  });

  std::vector<double> exp_pred(n);
  for (int i = 0; i < n; ++i) exp_pred[i] = safe_exp(pred[i]);

  std::vector<double> uniq_times;
  std::vector<std::vector<int>> blocks;
  for (int idx = 0; idx < n; ) {
    double t = time[order[idx]];
    std::vector<int> block;
    while (idx < n && time[order[idx]] == t) {
      block.push_back(order[idx]);
      ++idx;
    }
    uniq_times.push_back(t);
    blocks.push_back(std::move(block));
  }

  std::vector<double> risk_sum(blocks.size(), 0.0);
  double running = 0.0;
  for (int b = static_cast<int>(blocks.size()) - 1; b >= 0; --b) {
    for (int idx : blocks[b]) running += exp_pred[idx];
    risk_sum[b] = running;
  }

  double loss = 0.0;
  for (size_t b = 0; b < blocks.size(); ++b) {
    double log_denom = std::log(std::max(risk_sum[b], 1e-12));
    for (int idx : blocks[b]) {
      if (status[idx] == 1) loss += -(pred[idx] - log_denom);
    }
  }
  return loss;
}

// Negative AFT (normal location-scale) log-likelihood, mirroring the per-observation
// formulas in compute_aft_grad_hess. Used as the early-stopping metric. Drops the
// additive log(sigma) normalizing constant for events, since sigma is fixed for the
// whole run and that constant does not affect which round has the lowest loss.
static double compute_aft_loss(const NumericVector& pred, const NumericVector& time,
                               const IntegerVector& status, double sigma = 1.0) {
  int n = pred.size();
  double loss = 0.0;
  for (int i = 0; i < n; ++i) {
    double logt = std::log(time[i]);
    if (status[i] == 1) {
      double z = (pred[i] - logt) / sigma;
      loss += 0.5 * z * z;
    } else {
      double z = (logt - pred[i]) / sigma;
      double sf = std::max(1e-12, R::pnorm5(z, 0.0, 1.0, /*lower_tail=*/false, /*log_p=*/false));
      loss += -std::log(sf);
    }
  }
  return loss;
}

// Internal test hook: exposes the compiled gradient/Hessian formulas directly so R-level
// tests can verify them against finite-difference references without duplicating the C++
// math by hand. Not part of the public API.
extern "C" SEXP survgbm_grad_hess_cpp(SEXP predSEXP, SEXP timeSEXP, SEXP statusSEXP, SEXP objectiveSEXP) {
  BEGIN_RCPP
  NumericVector pred(predSEXP);
  NumericVector time(timeSEXP);
  IntegerVector status(statusSEXP);
  std::string objective = as<std::string>(objectiveSEXP);
  int n = pred.size();
  NumericVector grad(n), hess(n);
  if (objective == "aft") {
    compute_aft_grad_hess(pred, time, status, grad, hess, 1.0);
  } else if (objective == "pexp") {
    compute_pexp_grad_hess(pred, time, status, grad, hess);
  } else {
    compute_cox_grad_hess(pred, time, status, grad, hess);
  }
  return List::create(_["grad"] = grad, _["hess"] = hess);
  END_RCPP
}

extern "C" SEXP survgbm_fit_cpp(SEXP xSEXP, SEXP timeSEXP, SEXP statusSEXP,
                                SEXP objectiveSEXP, SEXP ntreesSEXP, SEXP learning_rateSEXP,
                                SEXP max_depthSEXP, SEXP min_node_sizeSEXP, SEXP max_binsSEXP,
                                SEXP subsampleSEXP, SEXP colsampleSEXP, SEXP lambdaSEXP,
                                SEXP gammaSEXP, SEXP min_child_weightSEXP, SEXP seedSEXP,
                                SEXP threadsSEXP, SEXP verboseSEXP,
                                SEXP x_validSEXP, SEXP time_validSEXP, SEXP status_validSEXP,
                                SEXP early_stoppingSEXP) {
  BEGIN_RCPP
  NumericMatrix x(xSEXP);
  NumericVector time(timeSEXP);
  IntegerVector status(statusSEXP);
  std::string objective = as<std::string>(objectiveSEXP);
  int ntrees = as<int>(ntreesSEXP);
  double learning_rate = as<double>(learning_rateSEXP);
  int max_depth = as<int>(max_depthSEXP);
  int min_node_size = as<int>(min_node_sizeSEXP);
  int max_bins = as<int>(max_binsSEXP);
  double subsample = as<double>(subsampleSEXP);
  double colsample = as<double>(colsampleSEXP);
  double lambda = as<double>(lambdaSEXP);
  double gamma = as<double>(gammaSEXP);
  double min_child_weight = as<double>(min_child_weightSEXP);
  int seed = as<int>(seedSEXP);
  int threads = as<int>(threadsSEXP);
  bool verbose = as<bool>(verboseSEXP);
  int early_stopping = as<int>(early_stoppingSEXP);

  NumericMatrix x_valid(x_validSEXP);
  NumericVector time_valid(time_validSEXP);
  IntegerVector status_valid(status_validSEXP);
  bool has_validation = x_valid.nrow() > 0 && early_stopping > 0;

  // `threads <= 0` ("automatic") is passed through to parallelFor() as -1, which
  // resolves to RcppParallel's hardware-concurrency default (or the
  // RCPP_PARALLEL_NUM_THREADS env var, if set).
  int num_threads = threads > 0 ? threads : -1;

  int n = x.nrow();
  int p = x.ncol();
  if (time.size() != n || status.size() != n) {
    stop("`time` and `status` must match `nrow(x)`.");
  }

  std::vector<std::vector<double>> cuts(p);
  std::vector<int> max_bin_by_feature(p, 1);
  for (int j = 0; j < p; ++j) {
    std::vector<double> col(n);
    for (int i = 0; i < n; ++i) col[i] = x(i, j);
    cuts[j] = make_cuts(col, max_bins);
    max_bin_by_feature[j] = static_cast<int>(cuts[j].size()) + 1;
  }

  IntegerMatrix bins(n, p);
  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < n; ++i) {
      bins(i, j) = bin_value(x(i, j), cuts[j]);
    }
  }

  // Validation set is binned with the *training* cuts (never its own), exactly like
  // prediction on new data -- it must never influence bin boundaries.
  int n_valid = has_validation ? x_valid.nrow() : 0;
  IntegerMatrix bins_valid(std::max(n_valid, 1), p);
  if (has_validation) {
    for (int j = 0; j < p; ++j) {
      for (int i = 0; i < n_valid; ++i) {
        bins_valid(i, j) = bin_value(x_valid(i, j), cuts[j]);
      }
    }
  }

  NumericVector pred(n);
  double init_score = 0.0;
  if (objective == "aft") {
    double sum_logt = 0.0;
    int count = 0;
    for (int i = 0; i < n; ++i) {
      if (status[i] == 1) {
        sum_logt += std::log(time[i]);
        ++count;
      }
    }
    if (count == 0) {
      for (int i = 0; i < n; ++i) sum_logt += std::log(time[i]);
      count = n;
    }
    init_score = sum_logt / std::max(1, count);
  } else if (objective == "pexp") {
    // Natural Poisson-regression intercept: log(total events / total exposure).
    double total_event = 0.0, total_exposure = 0.0;
    for (int i = 0; i < n; ++i) {
      total_event += status[i];
      total_exposure += time[i];
    }
    init_score = std::log(std::max(total_event, 1e-6) / std::max(total_exposure, 1e-12));
  }
  // Cox has no intercept in the partial likelihood: init_score stays 0.
  std::fill(pred.begin(), pred.end(), init_score);

  NumericVector pred_valid(std::max(n_valid, 1), init_score);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ur(0.0, 1.0);

  std::vector<TreeModel> trees;
  trees.reserve(ntrees);
  std::vector<double> importance(p, 0.0);

  std::vector<double> validation_history;
  validation_history.reserve(has_validation ? ntrees : 0);
  int best_iteration = 0;
  double best_val_loss = R_PosInf;
  int rounds_since_improvement = 0;
  std::string stopping_reason = has_validation ? "max_ntrees" : "disabled";

  for (int m = 0; m < ntrees; ++m) {
    NumericVector grad(n), hess(n);
    if (objective == "aft") {
      compute_aft_grad_hess(pred, time, status, grad, hess, 1.0);
    } else if (objective == "pexp") {
      compute_pexp_grad_hess(pred, time, status, grad, hess);
    } else {
      compute_cox_grad_hess(pred, time, status, grad, hess);
    }

    std::vector<int> rows;
    rows.reserve(n);
    for (int i = 0; i < n; ++i) {
      if (subsample >= 1.0 || ur(rng) < subsample) rows.push_back(i);
    }
    if (rows.empty()) {
      for (int i = 0; i < n; ++i) rows.push_back(i);
    }

    TreeModel tree = build_tree(bins, grad, hess, rows, max_depth, min_node_size, lambda, gamma, min_child_weight, p, colsample, rng, importance, num_threads);
    for (int i = 0; i < n; ++i) {
      pred[i] += learning_rate * predict_tree_row(tree, bins, i);
    }

    if (has_validation) {
      for (int i = 0; i < n_valid; ++i) {
        pred_valid[i] += learning_rate * predict_tree_row(tree, bins_valid, i);
      }
      double val_loss;
      if (objective == "aft") {
        val_loss = compute_aft_loss(pred_valid, time_valid, status_valid, 1.0);
      } else if (objective == "pexp") {
        val_loss = compute_pexp_loss(pred_valid, time_valid, status_valid);
      } else {
        val_loss = compute_cox_loss(pred_valid, time_valid, status_valid);
      }
      validation_history.push_back(val_loss);
      if (val_loss < best_val_loss) {
        best_val_loss = val_loss;
        best_iteration = m + 1;
        rounds_since_improvement = 0;
      } else {
        ++rounds_since_improvement;
      }
    }

    trees.push_back(std::move(tree));

    if (verbose && ((m + 1) % 50 == 0 || m + 1 == ntrees)) {
      Rcout << "trained tree " << (m + 1) << "/" << ntrees << "\n";
    }

    if (has_validation && rounds_since_improvement >= early_stopping) {
      stopping_reason = "early_stopping";
      break;
    }
  }

  List tree_list(trees.size());
  for (size_t i = 0; i < trees.size(); ++i) {
    tree_list[i] = make_tree_list(trees[i]);
  }

  // A base R matrix's column names live at dimnames(x)[[2]], not a top-level
  // "colnames" attribute (that's only what the colnames() accessor computes).
  CharacterVector feature_names;
  SEXP dimnames_attr = x.attr("dimnames");
  if (dimnames_attr != R_NilValue) {
    List dimnames(dimnames_attr);
    if (dimnames.size() == 2 && dimnames[1] != R_NilValue) {
      feature_names = dimnames[1];
    }
  }
  if (feature_names.size() != p) {
    feature_names = CharacterVector(p);
    for (int j = 0; j < p; ++j) feature_names[j] = "V" + std::to_string(j + 1);
  }

  NumericVector importance_vec(p);
  for (int j = 0; j < p; ++j) importance_vec[j] = importance[j];

  return List::create(
    _["objective"] = objective,
    _["init_score"] = init_score,
    _["trees"] = tree_list,
    _["feature_names"] = feature_names,
    _["feature_importance"] = importance_vec,
    _["cuts"] = cuts,
    _["max_bin_by_feature"] = max_bin_by_feature,
    _["learning_rate"] = learning_rate,
    _["max_depth"] = max_depth,
    _["min_node_size"] = min_node_size,
    _["max_bins"] = max_bins,
    _["survival_sigma"] = 1.0,
    _["best_iteration"] = has_validation ? best_iteration : static_cast<int>(trees.size()),
    _["validation_history"] = wrap(validation_history),
    _["stopping_reason"] = stopping_reason
  );
  END_RCPP
}

extern "C" SEXP survgbm_predict_cpp(SEXP modelSEXP, SEXP xSEXP, SEXP typeSEXP) {
  BEGIN_RCPP
  (void)typeSEXP;  // reserved for future prediction-type variants (e.g. per-tree contributions)
  List model(modelSEXP);
  NumericMatrix x(xSEXP);
  double init_score = as<double>(model["init_score"]);
  List cuts_list(model["cuts"]);
  List trees_list(model["trees"]);
  double learning_rate = as<double>(model["learning_rate"]);
  int n = x.nrow();
  int p = x.ncol();
  IntegerMatrix bins(n, p);
  for (int j = 0; j < p; ++j) {
    std::vector<double> cuts = as<std::vector<double>>(cuts_list[j]);
    for (int i = 0; i < n; ++i) {
      bins(i, j) = bin_value(x(i, j), cuts);
    }
  }
  NumericVector pred(n, init_score);
  for (int t = 0; t < trees_list.size(); ++t) {
    List tree_list(trees_list[t]);
    IntegerVector feature = tree_list["feature"];
    IntegerVector threshold = tree_list["threshold"];
    LogicalVector missing_left = tree_list["missing_left"];
    IntegerVector left = tree_list["left"];
    IntegerVector right = tree_list["right"];
    NumericVector value = tree_list["value"];
    for (int i = 0; i < n; ++i) {
      int node = 0;
      while (true) {
        if (feature[node] < 0) {
          pred[i] += learning_rate * value[node];
          break;
        }
        int b = bins(i, feature[node]);
        if (b == 0) {
          node = missing_left[node] ? left[node] : right[node];
        } else if (b <= threshold[node]) {
          node = left[node];
        } else {
          node = right[node];
        }
      }
    }
  }
  return pred;
  END_RCPP
}

extern "C" SEXP survgbm_importance_cpp(SEXP modelSEXP) {
  BEGIN_RCPP
  List model(modelSEXP);
  NumericVector imp = model["feature_importance"];
  return imp;
  END_RCPP
}
