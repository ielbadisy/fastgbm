#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

using namespace Rcpp;

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
  double left_value = 0.0;
  double right_value = 0.0;
};

static inline double sigmoid(double x) {
  if (x > 35.0) return 1.0;
  if (x < -35.0) return 0.0;
  return 1.0 / (1.0 + std::exp(-x));
}

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

static TreeModel build_tree(const IntegerMatrix& bins, const NumericVector& grad,
                            const NumericVector& hess, const std::vector<int>& rows,
                            int max_depth, int min_node_size, double lambda,
                            double gamma, double min_child_weight,
                            const std::vector<int>& feature_ids,
                            std::vector<double>& importance) {
  TreeModel tree;
  tree.nodes.reserve(2 * rows.size() + 1);

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

    NodeSplit best;
    for (int feature : feature_ids) {
      int max_bin = 0;
      for (int r : node_rows) {
        int b = bins(r, feature);
        if (b > max_bin) max_bin = b;
      }
      if (max_bin < 2) continue;

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

      double left_g = 0.0;
      double left_h = 0.0;
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
          if (gain > best.gain) {
            best.feature = feature;
            best.threshold = t;
            best.missing_left = (missing_side == 0);
            best.gain = gain;
          }
        }
      }
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

static void compute_regression_grad_hess(const NumericVector& pred, const NumericVector& y,
                                         NumericVector& grad, NumericVector& hess) {
  int n = pred.size();
  for (int i = 0; i < n; ++i) {
    grad[i] = pred[i] - y[i];
    hess[i] = 1.0;
  }
}

static void compute_binary_grad_hess(const NumericVector& pred, const NumericVector& y,
                                     NumericVector& grad, NumericVector& hess) {
  int n = pred.size();
  for (int i = 0; i < n; ++i) {
    double p_i = sigmoid(pred[i]);
    grad[i] = p_i - y[i];
    hess[i] = std::max(1e-6, p_i * (1.0 - p_i));
  }
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
      double z = (logt - pred[i]) / sigma;
      double sf = std::max(1e-12, R::pnorm5(z, 0.0, 1.0, /*lower_tail=*/false, /*log_p=*/false));
      double pdf = R::dnorm4(z, 0.0, 1.0, /*give_log=*/false);
      grad[i] = pdf / (sigma * sf);
      double h = pdf * (z * sf - pdf) / (sigma2 * sf * sf);
      hess[i] = std::max(1e-6, h);
    }
  }
}

extern "C" SEXP fastgbm_fit_cpp(SEXP xSEXP, SEXP ySEXP, SEXP timeSEXP, SEXP statusSEXP,
                                SEXP objectiveSEXP, SEXP ntreesSEXP, SEXP learning_rateSEXP,
                                SEXP max_depthSEXP, SEXP min_node_sizeSEXP, SEXP max_binsSEXP,
                                SEXP subsampleSEXP, SEXP colsampleSEXP, SEXP lambdaSEXP,
                                SEXP gammaSEXP, SEXP min_child_weightSEXP, SEXP seedSEXP,
                                SEXP verboseSEXP) {
  BEGIN_RCPP
  NumericMatrix x(xSEXP);
  NumericVector y(ySEXP);
  NumericVector time = timeSEXP == R_NilValue ? NumericVector() : NumericVector(timeSEXP);
  IntegerVector status = statusSEXP == R_NilValue ? IntegerVector() : IntegerVector(statusSEXP);
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
  bool verbose = as<bool>(verboseSEXP);

  int n = x.nrow();
  int p = x.ncol();
  bool survival_objective = (objective == "survival:cox" || objective == "survival:aft");
  if (survival_objective) {
    if (time.size() != n || status.size() != n) {
      stop("`time` and `status` are required for survival objectives and must match `x`.");
    }
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

  NumericVector pred(n);
  double init_score = 0.0;
  auto mean_vec = [&](const NumericVector& v) {
    double s = 0.0;
    for (double val : v) s += val;
    return s / std::max<R_xlen_t>(1, v.size());
  };
  if (objective == "binary:logistic") {
    double mean_y = std::min(1.0 - 1e-6, std::max(1e-6, mean_vec(y)));
    init_score = std::log(mean_y / (1.0 - mean_y));
  } else if (objective == "survival:aft") {
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
  } else {
    init_score = mean_vec(y);
  }
  std::fill(pred.begin(), pred.end(), init_score);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ur(0.0, 1.0);

  std::vector<TreeModel> trees;
  trees.reserve(ntrees);
  std::vector<double> importance(p, 0.0);

  for (int m = 0; m < ntrees; ++m) {
    NumericVector grad(n), hess(n);
    if (objective == "binary:logistic") {
      compute_binary_grad_hess(pred, y, grad, hess);
    } else if (objective == "survival:cox") {
      compute_cox_grad_hess(pred, time, status, grad, hess);
    } else if (objective == "survival:aft") {
      compute_aft_grad_hess(pred, time, status, grad, hess, 1.0);
    } else {
      compute_regression_grad_hess(pred, y, grad, hess);
    }

    std::vector<int> rows;
    rows.reserve(n);
    for (int i = 0; i < n; ++i) {
      if (subsample >= 1.0 || ur(rng) < subsample) rows.push_back(i);
    }
    if (rows.empty()) {
      for (int i = 0; i < n; ++i) rows.push_back(i);
    }

    std::vector<int> feature_ids;
    feature_ids.reserve(p);
    for (int j = 0; j < p; ++j) {
      if (colsample >= 1.0 || ur(rng) < colsample) feature_ids.push_back(j);
    }
    if (feature_ids.empty()) {
      for (int j = 0; j < p; ++j) feature_ids.push_back(j);
    }

    TreeModel tree = build_tree(bins, grad, hess, rows, max_depth, min_node_size, lambda, gamma, min_child_weight, feature_ids, importance);
    for (int i = 0; i < n; ++i) {
      pred[i] += learning_rate * predict_tree_row(tree, bins, i);
    }
    trees.push_back(std::move(tree));

    if (verbose && ((m + 1) % 50 == 0 || m + 1 == ntrees)) {
      Rcout << "trained tree " << (m + 1) << "/" << ntrees << "\n";
    }
  }

  List tree_list(trees.size());
  for (size_t i = 0; i < trees.size(); ++i) {
    tree_list[i] = make_tree_list(trees[i]);
  }

  CharacterVector feature_names;
  SEXP colnames_attr = x.attr("colnames");
  if (colnames_attr != R_NilValue) {
    feature_names = colnames_attr;
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
    _["survival_sigma"] = 1.0
  );
  END_RCPP
}

extern "C" SEXP fastgbm_predict_cpp(SEXP modelSEXP, SEXP xSEXP, SEXP typeSEXP) {
  BEGIN_RCPP
  List model(modelSEXP);
  NumericMatrix x(xSEXP);
  std::string type = as<std::string>(typeSEXP);
  std::string objective = as<std::string>(model["objective"]);
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
    int nodes = feature.size();
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

  if (type == "response" && objective == "binary:logistic") {
    for (int i = 0; i < n; ++i) pred[i] = sigmoid(pred[i]);
  }
  return pred;
  END_RCPP
}

extern "C" SEXP fastgbm_importance_cpp(SEXP modelSEXP) {
  BEGIN_RCPP
  List model(modelSEXP);
  NumericVector imp = model["feature_importance"];
  return imp;
  END_RCPP
}
