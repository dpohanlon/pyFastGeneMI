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
#include <armadillo>
#include <iostream>
#include "omp.h"

// ----------------------------------------------------------------------------------
//  Maximum Likelihood, Miller-Madow, Chao-Shen and Shrinkage Mutual Information
//  Estimators
// ----------------------------------------------------------------------------------

// Maximum likelihood mutual information
arma::mat mim_ML_cpp(const arma::mat& disc_expr_data, int n_cores)
{
  // Convert from assumed external 1-indexing to C++ indexing
  arma::Mat<int> data = arma::conv_to<arma::Mat<int> >::from(disc_expr_data - 1.0);
  const int n_genes(data.n_cols), n_samples(data.n_rows);
  const int n_pairs = get_n_gene_pairs(n_genes);

  // Compute marginal entropies
  std::vector<double> h_marginals(n_genes);
  for(int j = 0; j < n_genes; ++j)
  {
    arma::vec p_marginal = get_emp_marg_dist(data.col(j));
    h_marginals[j] = get_marginal_ml_entropy(p_marginal);
  }

  // Compute joint entropies in parallel
  std::vector<double> h_joints(n_pairs); // Change to armadillo vector for returning to caller

  // Set the number of cores to use
  omp_set_num_threads(n_cores);

  const std::vector<std::pair<int,int>> ij_pairs = get_ij_list(n_genes);

  #pragma omp parallel for shared(h_joints)
  for(int ij = 0; ij < n_pairs; ++ij)
  {
    std::pair<int,int> ij_pair = ij_pairs[ij];
    int i = ij_pair.first, j = ij_pair.second;
    arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));
    h_joints[ij] = get_joint_ml_entropy(p_joint);
  }

  // Compute mutual information
  arma::mat mim = arma::mat(n_genes, n_genes, arma::fill::zeros);
  int ij = 0;
  for(int i = 0; i < n_genes; ++i)
  {
    for(int j = i; j < n_genes; ++j)
    {
      mim(i,j) = h_marginals[i] + h_marginals[j] - h_joints[ij];
      mim(j,i) = mim(i,j);
      ++ij;
    }
  }

  return mim;
}


// Mutual information using maximum likelihood entropy estimate and
// Miller-Madow bias correction
arma::mat mim_MM_cpp(const arma::mat& disc_expr_data, int n_cores)
{
  arma::Mat<int> data = arma::conv_to<arma::Mat<int> >::from(disc_expr_data - 1.0);
  const int n_genes(data.n_cols), n_samples(data.n_rows);
  const int n_pairs = get_n_gene_pairs(n_genes);

  // Compute marginal entropies
  std::vector<double> h_marginals(n_genes);
  for(int j = 0; j < n_genes; ++j)
  {
    // Compute the Miller-Madow correction to the entropy
    arma::vec p_marginal = get_emp_marg_dist(data.col(j));
    int nonzero_bins = arma::size(arma::find(p_marginal))(0);
    double mm_corr = static_cast<double>(nonzero_bins - 1) / (2.0 * static_cast<double>(n_samples));

    h_marginals[j] = get_marginal_ml_entropy(p_marginal) + mm_corr;
  }

  // Compute joint entropies in parallel
  std::vector<double> h_joints(n_pairs);

  // Set the number of cores to use
  omp_set_num_threads(n_cores);

  const std::vector<std::pair<int,int>> ij_pairs = get_ij_list(n_genes);

  #pragma omp parallel for shared(h_joints)
  for(int ij = 0; ij < n_pairs; ++ij)
  {
    std::pair<int,int> ij_pair = ij_pairs[ij];
    int i = ij_pair.first, j = ij_pair.second;
    arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));
    int nonzero_bins = arma::size(arma::find(p_joint))(0);
    double mm_corr = static_cast<double>(nonzero_bins - 1) / (2.0 * static_cast<double>(n_samples));
    h_joints[ij] = get_joint_ml_entropy(p_joint) + mm_corr;
  }

  // Compute mutual information
  arma::mat mim = arma::mat(n_genes, n_genes, arma::fill::zeros);
  int ij = 0;
  for(int i = 0; i < n_genes; ++i)
  {
    for(int j = i; j < n_genes; ++j)
    {
      mim(i,j) = h_marginals[i] + h_marginals[j] - h_joints[ij];
      mim(j,i) = mim(i,j);
      ++ij;
    }
  }

  // Zero negative values
  mim.elem( arma::find(mim < 0.0) ).fill(0.0);

  return mim;
}


// Chao-Shen Estimator
arma::mat mim_CS_cpp(const arma::mat& disc_expr_data, int n_cores)
{
  arma::Mat<int> data = arma::conv_to<arma::Mat<int> >::from(disc_expr_data - 1.0);
  const int n_genes(data.n_cols), n_samples(data.n_rows);
  const int n_pairs = get_n_gene_pairs(n_genes);

  // Compute marginal entropies
  std::vector<double> h_marginals(n_genes);
  for(int j = 0; j < n_genes; ++j)
  {
    // The number of bins with a single count
    arma::vec p_marginal = get_emp_marg_dist(data.col(j));
    int sing_count_bins = arma::size(arma::find(p_marginal == 1.0 / static_cast<double>(n_samples)))(0);
    double samp_cov = 1.0 - static_cast<double>(sing_count_bins) / static_cast<double>(n_samples);
    arma::vec cs_corr = 1.0 / (1.0 - arma::pow(1.0 - samp_cov * p_marginal, n_samples));

    // Set inf for empty bins to zero
    cs_corr.elem( arma::find(cs_corr == arma::datum::inf) ).zeros();

    h_marginals[j] = -arma::sum(samp_cov * p_marginal % arma::log(samp_cov * p_marginal + 1e-16) % cs_corr);
  }

  // Compute joint entropies in parallel
  std::vector<double> h_joints(n_pairs);

  // Set the number of cores to use
  omp_set_num_threads(n_cores);

  const std::vector<std::pair<int,int>> ij_pairs = get_ij_list(n_genes);

  #pragma omp parallel for shared(h_joints)
  for(int ij = 0; ij < n_pairs; ++ij)
  {
    std::pair<int,int> ij_pair = ij_pairs[ij];
    int i = ij_pair.first, j = ij_pair.second;
    arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));
    int sing_count_bins = arma::size(arma::find(p_joint == 1.0 / static_cast<double>(n_samples)))(0);
    double samp_cov = 1.0 - static_cast<double>(sing_count_bins) / static_cast<double>(n_samples);
    arma::mat cs_corr = 1.0 / (1.0 - arma::pow(1.0 - samp_cov * p_joint, n_samples));

    // Set inf from division by zero to zero
    cs_corr.elem( arma::find(cs_corr == arma::datum::inf) ).zeros();

    arma::mat tmp = samp_cov * p_joint % arma::log(samp_cov * p_joint + 1e-16) % cs_corr;
    h_joints[ij] = -arma::accu(tmp);
  }

  // Compute mutual information
  arma::mat mim = arma::mat(n_genes, n_genes, arma::fill::zeros);
  int ij = 0;
  for(int i = 0; i < n_genes; ++i)
  {
    for(int j = i; j < n_genes; ++j)
    {
      mim(i,j) = h_marginals[i] + h_marginals[j] - h_joints[ij];
      mim(j,i) = mim(i,j);
      ++ij;
    }
  }

  // Zero negative values
  mim.elem( arma::find(mim < 0.0) ).fill(0.0);

  return mim;
}


// Shrinkage estimator
arma::mat mim_shrink_cpp(const arma::mat& disc_expr_data, int n_cores)
{
  arma::Mat<int> data = arma::conv_to<arma::Mat<int> >::from(disc_expr_data - 1.0);   // Change from assumed R indexing to C++ indexing
  const int n_genes(data.n_cols), n_samples(data.n_rows);
  const int n_pairs = get_n_gene_pairs(n_genes);

  // Compute marginal entropies
  std::vector<double> h_marginals(n_genes);
  for(int j = 0; j < n_genes; ++j)
  {
    // Compute the shrinkage intensity lambda
    arma::vec p_marginal = get_emp_marg_dist(data.col(j));
    double n_bins = static_cast<double>(p_marginal.n_elem);
    double lambda_numer = 1.0 - arma::accu(arma::pow(p_marginal, 2.0));
    double lambda_denom = static_cast<double>(n_samples - 1) *
      arma::accu(arma::pow(1.0 / n_bins - p_marginal, 2.0));

    double lambda;
    if(lambda_denom == 0)
      lambda = 0;
    else
      lambda = lambda_numer / lambda_denom;

    // Lambda must be between 0 and 1
    if(lambda < 0.0)
      lambda = 0.0;
    else if(lambda > 1)
      lambda = 1.0;

    // Note: the target distribution is uniform
    arma::vec p_marg_shrink = lambda * (1.0 / n_bins) + (1.0 - lambda) * p_marginal;
    h_marginals[j] = get_marginal_ml_entropy(p_marg_shrink);
  }

  // Compute joint entropies in parallel
  std::vector<double> h_joints(n_pairs);

  // Set the number of cores to use
  omp_set_num_threads(n_cores);

  const std::vector<std::pair<int,int>> ij_pairs = get_ij_list(n_genes);

  #pragma omp parallel for shared(h_joints)
  for(int ij = 0; ij < n_pairs; ++ij)
  {
    std::pair<int,int> ij_pair = ij_pairs[ij];
    int i = ij_pair.first, j = ij_pair.second;
    arma::mat p_joint = get_emp_joint_dist(data.col(i), data.col(j));
    double n_bins = static_cast<double>(p_joint.n_elem);
    double lambda_numer = 1.0 - arma::accu(arma::pow(p_joint, 2.0));
    double lambda_denom = static_cast<double>(n_samples - 1) *
          arma::accu(arma::pow(1.0 / n_bins - p_joint, 2.0));

    double lambda;
    if(lambda_denom == 0.0)
      lambda = 0.0;
    else
      lambda = lambda_numer / lambda_denom;

    // Lambda must be between 0 and 1
    if(lambda < 0.0)
      lambda = 0.0;
    else if(lambda > 1)
      lambda = 1.0;

    // Note: the target distribution is uniform
    arma::mat p_joint_shrink = lambda * (1.0 / n_bins) + (1.0 - lambda) * p_joint;
    h_joints[ij] = get_joint_ml_entropy(p_joint_shrink);
  }

  // Compute mutual information
  arma::mat mim = arma::mat(n_genes, n_genes, arma::fill::zeros);
  int ij = 0;
  for(int i = 0; i < n_genes; ++i)
  {
    for(int j = i; j < n_genes; ++j)
    {
      // Uncomment the following lines for debugging output if desired
      // std::cout << "i=" << i << " j=" << j << std::endl;
      // std::cout << "h.i = " << h_marginals[i] << " h.j = " << h_marginals[j] << std::endl;
      // std::cout << "h.ij = " << h_joints[ij] << std::endl << std::endl;

      mim(i,j) = h_marginals[i] + h_marginals[j] - h_joints[ij];
      mim(j,i) = mim(i,j);
      ++ij;
    }
  }

  return mim;
}
