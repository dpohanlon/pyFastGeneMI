#ifndef __FAST_MINET__
#define __FAST_MINET__

#include <armadillo>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <utility>

#include <omp.h>
#include <unistd.h>

#include "utils.hpp"
#include "empirical_dist.hpp"

// In disc_mi_estimators.cpp
arma::mat mim_ML_cpp(const arma::mat& disc_expr_data);
arma::mat mim_MM_cpp(const arma::mat& disc_expr_data);
arma::mat mim_CS_cpp(const arma::mat& disc_expr_data);
arma::mat mim_shrink_cpp(const arma::mat& disc_expr_data);

// In emprical_dist_getters.cpp
arma::vec get_emp_marg_dist(const arma::Mat<int>& disc_data);
arma::mat get_emp_joint_dist(const arma::Mat<int>& disc_data_col_i, const arma::Mat<int>& disc_data_col_j);

// In util_funcs.cpp
double get_marginal_ml_entropy(const arma::vec& p_marg);
double get_joint_ml_entropy(const arma::mat& p_joint);
int get_n_gene_pairs(const int n_genes);
arma::Mat<int> get_idx_lookup_mat(const int n_genes);
std::vector<std::pair<int,int> > get_ij_list(const int n_genes);

#endif // __FAST_MINET__
