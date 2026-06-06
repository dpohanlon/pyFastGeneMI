// MIT License

// Copyright (c) 2018 Jonathan Ish-Horowicz

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// ----------------------------------------------------------------------------------
//  Implementations of the Mutual Information estimators using Maximum likelihood,
//  Miller-Madow, Chao-Shen, Jack-knififed (unvalidated) and Shrinkage entropy
//  estimation (problem in joint entropy computation). For references see
//  documentation
// ----------------------------------------------------------------------------------
#include "fastGeneMI.h"
#include "utils.hpp"
#include "empirical_dist.hpp"

#include <algorithm>
#include <armadillo>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

inline void validate_discrete_matrix(const arma::Mat<int>& data)
{
  if(data.n_rows == 0 || data.n_cols == 0)
  {
    throw std::invalid_argument("disc_expr_data must be a non-empty 2D matrix");
  }

  if(data.min() < 0)
  {
    throw std::invalid_argument("disc_expr_data contains negative values; discrete labels must be >= 0");
  }
}

inline int sanitize_n_cores(int n_cores)
{
#ifdef _OPENMP
  if(n_cores <= 1)
  {
    return 1;
  }
  return std::min(n_cores, omp_get_max_threads());
#else
  return 1;
#endif
}

inline arma::mat mim_ML_cpp(const arma::Mat<int>& data, int n_cores)
{
  validate_discrete_matrix(data);

  const int n_genes = static_cast<int>(data.n_cols);
  const int n_pairs = get_n_gene_pairs(n_genes);
  const int n_threads = sanitize_n_cores(n_cores);

  std::vector<double> h_marginals(n_genes);
  for(int j = 0; j < n_genes; ++j)
  {
    const arma::Col<int> col = data.col(j);
    arma::vec p_marginal = get_emp_marg_dist(col);
    h_marginals[j] = get_marginal_ml_entropy(p_marginal);
  }

  std::vector<double> h_joints(n_pairs);
  const std::vector<std::pair<int, int>> ij_pairs = get_ij_list(n_genes);

#ifdef _OPENMP
  if(n_threads > 1)
  {
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(static)
    for(int ij = 0; ij < n_pairs; ++ij)
    {
      const int i = ij_pairs[ij].first;
      const int j = ij_pairs[ij].second;
      arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));
      h_joints[ij] = get_joint_ml_entropy(p_joint);
    }
  }
  else
#endif
  {
    for(int ij = 0; ij < n_pairs; ++ij)
    {
      const int i = ij_pairs[ij].first;
      const int j = ij_pairs[ij].second;
      arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));
      h_joints[ij] = get_joint_ml_entropy(p_joint);
    }
  }

  arma::mat mim(n_genes, n_genes, arma::fill::zeros);
  int ij = 0;
  for(int i = 0; i < n_genes; ++i)
  {
    for(int j = i; j < n_genes; ++j)
    {
      mim(i, j) = h_marginals[i] + h_marginals[j] - h_joints[ij];
      mim(j, i) = mim(i, j);
      ++ij;
    }
  }

  return mim;
}

inline arma::mat mim_MM_cpp(const arma::Mat<int>& data, int n_cores)
{
  validate_discrete_matrix(data);

  const int n_genes = static_cast<int>(data.n_cols);
  const int n_samples = static_cast<int>(data.n_rows);
  const int n_pairs = get_n_gene_pairs(n_genes);
  const int n_threads = sanitize_n_cores(n_cores);

  std::vector<double> h_marginals(n_genes);
  for(int j = 0; j < n_genes; ++j)
  {
    const arma::Col<int> col = data.col(j);
    arma::vec p_marginal = get_emp_marg_dist(col);

    int nonzero_bins = static_cast<int>(arma::accu(p_marginal > 0.0));
    double mm_corr = static_cast<double>(nonzero_bins - 1) / (2.0 * static_cast<double>(n_samples));

    h_marginals[j] = get_marginal_ml_entropy(p_marginal) + mm_corr;
  }

  std::vector<double> h_joints(n_pairs);
  const std::vector<std::pair<int, int>> ij_pairs = get_ij_list(n_genes);

#ifdef _OPENMP
  if(n_threads > 1)
  {
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(static)
    for(int ij = 0; ij < n_pairs; ++ij)
    {
      const int i = ij_pairs[ij].first;
      const int j = ij_pairs[ij].second;
      arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));

      int nonzero_bins = static_cast<int>(arma::accu(p_joint > 0.0));
      double mm_corr = static_cast<double>(nonzero_bins - 1) / (2.0 * static_cast<double>(n_samples));
      h_joints[ij] = get_joint_ml_entropy(p_joint) + mm_corr;
    }
  }
  else
#endif
  {
    for(int ij = 0; ij < n_pairs; ++ij)
    {
      const int i = ij_pairs[ij].first;
      const int j = ij_pairs[ij].second;
      arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));

      int nonzero_bins = static_cast<int>(arma::accu(p_joint > 0.0));
      double mm_corr = static_cast<double>(nonzero_bins - 1) / (2.0 * static_cast<double>(n_samples));
      h_joints[ij] = get_joint_ml_entropy(p_joint) + mm_corr;
    }
  }

  arma::mat mim(n_genes, n_genes, arma::fill::zeros);
  int ij = 0;
  for(int i = 0; i < n_genes; ++i)
  {
    for(int j = i; j < n_genes; ++j)
    {
      mim(i, j) = h_marginals[i] + h_marginals[j] - h_joints[ij];
      mim(j, i) = mim(i, j);
      ++ij;
    }
  }

  mim.elem(arma::find(mim < 0.0)).fill(0.0);

  return mim;
}

inline arma::mat mim_CS_cpp(const arma::Mat<int>& data, int n_cores)
{
  validate_discrete_matrix(data);

  const int n_genes = static_cast<int>(data.n_cols);
  const int n_samples = static_cast<int>(data.n_rows);
  const int n_pairs = get_n_gene_pairs(n_genes);
  const int n_threads = sanitize_n_cores(n_cores);

  std::vector<double> h_marginals(n_genes);
  for(int j = 0; j < n_genes; ++j)
  {
    const arma::Col<int> col = data.col(j);
    arma::vec p_marginal = get_emp_marg_dist(col);

    int sing_count_bins = static_cast<int>(arma::accu(p_marginal == 1.0 / static_cast<double>(n_samples)));
    double samp_cov = 1.0 - static_cast<double>(sing_count_bins) / static_cast<double>(n_samples);
    arma::vec cs_corr = 1.0 / (1.0 - arma::pow(1.0 - samp_cov * p_marginal, n_samples));

    cs_corr.elem(arma::find_nonfinite(cs_corr)).zeros();

    h_marginals[j] = -arma::sum(samp_cov * p_marginal % arma::log(samp_cov * p_marginal + 1e-16) % cs_corr);
  }

  std::vector<double> h_joints(n_pairs);
  const std::vector<std::pair<int, int>> ij_pairs = get_ij_list(n_genes);

#ifdef _OPENMP
  if(n_threads > 1)
  {
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(static)
    for(int ij = 0; ij < n_pairs; ++ij)
    {
      const int i = ij_pairs[ij].first;
      const int j = ij_pairs[ij].second;
      arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));

      int sing_count_bins = static_cast<int>(arma::accu(p_joint == 1.0 / static_cast<double>(n_samples)));
      double samp_cov = 1.0 - static_cast<double>(sing_count_bins) / static_cast<double>(n_samples);
      arma::mat cs_corr = 1.0 / (1.0 - arma::pow(1.0 - samp_cov * p_joint, n_samples));

      cs_corr.elem(arma::find_nonfinite(cs_corr)).zeros();

      arma::mat tmp = samp_cov * p_joint % arma::log(samp_cov * p_joint + 1e-16) % cs_corr;
      h_joints[ij] = -arma::accu(tmp);
    }
  }
  else
#endif
  {
    for(int ij = 0; ij < n_pairs; ++ij)
    {
      const int i = ij_pairs[ij].first;
      const int j = ij_pairs[ij].second;
      arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));

      int sing_count_bins = static_cast<int>(arma::accu(p_joint == 1.0 / static_cast<double>(n_samples)));
      double samp_cov = 1.0 - static_cast<double>(sing_count_bins) / static_cast<double>(n_samples);
      arma::mat cs_corr = 1.0 / (1.0 - arma::pow(1.0 - samp_cov * p_joint, n_samples));

      cs_corr.elem(arma::find_nonfinite(cs_corr)).zeros();

      arma::mat tmp = samp_cov * p_joint % arma::log(samp_cov * p_joint + 1e-16) % cs_corr;
      h_joints[ij] = -arma::accu(tmp);
    }
  }

  arma::mat mim(n_genes, n_genes, arma::fill::zeros);
  int ij = 0;
  for(int i = 0; i < n_genes; ++i)
  {
    for(int j = i; j < n_genes; ++j)
    {
      mim(i, j) = h_marginals[i] + h_marginals[j] - h_joints[ij];
      mim(j, i) = mim(i, j);
      ++ij;
    }
  }

  mim.elem(arma::find(mim < 0.0)).fill(0.0);

  return mim;
}

inline arma::mat mim_shrink_cpp(const arma::Mat<int>& data, int n_cores)
{
  validate_discrete_matrix(data);

  const int n_genes = static_cast<int>(data.n_cols);
  const int n_samples = static_cast<int>(data.n_rows);
  const int n_pairs = get_n_gene_pairs(n_genes);
  const int n_threads = sanitize_n_cores(n_cores);

  std::vector<double> h_marginals(n_genes);
  for(int j = 0; j < n_genes; ++j)
  {
    const arma::Col<int> col = data.col(j);
    arma::vec p_marginal = get_emp_marg_dist(col);

    double n_bins = static_cast<double>(p_marginal.n_elem);
    double lambda_numer = 1.0 - arma::accu(arma::pow(p_marginal, 2.0));
    double lambda_denom = static_cast<double>(n_samples - 1) *
      arma::accu(arma::pow(1.0 / n_bins - p_marginal, 2.0));

    double lambda = 0.0;
    if(lambda_denom != 0.0)
    {
      lambda = lambda_numer / lambda_denom;
    }

    if(lambda < 0.0)
    {
      lambda = 0.0;
    }
    else if(lambda > 1.0)
    {
      lambda = 1.0;
    }

    arma::vec p_marg_shrink = lambda * (1.0 / n_bins) + (1.0 - lambda) * p_marginal;
    h_marginals[j] = get_marginal_ml_entropy(p_marg_shrink);
  }

  std::vector<double> h_joints(n_pairs);
  const std::vector<std::pair<int, int>> ij_pairs = get_ij_list(n_genes);

#ifdef _OPENMP
  if(n_threads > 1)
  {
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(static)
    for(int ij = 0; ij < n_pairs; ++ij)
    {
      const int i = ij_pairs[ij].first;
      const int j = ij_pairs[ij].second;
      arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));

      double n_bins = static_cast<double>(p_joint.n_elem);
      double lambda_numer = 1.0 - arma::accu(arma::pow(p_joint, 2.0));
      double lambda_denom = static_cast<double>(n_samples - 1) *
        arma::accu(arma::pow(1.0 / n_bins - p_joint, 2.0));

      double lambda = 0.0;
      if(lambda_denom != 0.0)
      {
        lambda = lambda_numer / lambda_denom;
      }

      if(lambda < 0.0)
      {
        lambda = 0.0;
      }
      else if(lambda > 1.0)
      {
        lambda = 1.0;
      }

      arma::mat p_joint_shrink = lambda * (1.0 / n_bins) + (1.0 - lambda) * p_joint;
      h_joints[ij] = get_joint_ml_entropy(p_joint_shrink);
    }
  }
  else
#endif
  {
    for(int ij = 0; ij < n_pairs; ++ij)
    {
      const int i = ij_pairs[ij].first;
      const int j = ij_pairs[ij].second;
      arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));

      double n_bins = static_cast<double>(p_joint.n_elem);
      double lambda_numer = 1.0 - arma::accu(arma::pow(p_joint, 2.0));
      double lambda_denom = static_cast<double>(n_samples - 1) *
        arma::accu(arma::pow(1.0 / n_bins - p_joint, 2.0));

      double lambda = 0.0;
      if(lambda_denom != 0.0)
      {
        lambda = lambda_numer / lambda_denom;
      }

      if(lambda < 0.0)
      {
        lambda = 0.0;
      }
      else if(lambda > 1.0)
      {
        lambda = 1.0;
      }

      arma::mat p_joint_shrink = lambda * (1.0 / n_bins) + (1.0 - lambda) * p_joint;
      h_joints[ij] = get_joint_ml_entropy(p_joint_shrink);
    }
  }

  arma::mat mim(n_genes, n_genes, arma::fill::zeros);
  int ij = 0;
  for(int i = 0; i < n_genes; ++i)
  {
    for(int j = i; j < n_genes; ++j)
    {
      mim(i, j) = h_marginals[i] + h_marginals[j] - h_joints[ij];
      mim(j, i) = mim(i, j);
      ++ij;
    }
  }

  return mim;
}
