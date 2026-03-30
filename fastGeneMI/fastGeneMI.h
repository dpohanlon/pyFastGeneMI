#ifndef __FAST_MINET__
#define __FAST_MINET__

#include <armadillo>
#include <utility>
#include <vector>

arma::mat mim_ML_cpp(const arma::Mat<int>& disc_expr_data, int n_cores);
arma::mat mim_MM_cpp(const arma::Mat<int>& disc_expr_data, int n_cores);
arma::mat mim_CS_cpp(const arma::Mat<int>& disc_expr_data, int n_cores);
arma::mat mim_shrink_cpp(const arma::Mat<int>& disc_expr_data, int n_cores);

arma::vec get_emp_marg_dist(const arma::Col<int>& disc_data);
arma::mat get_emp_joint_dist(const arma::Col<int>& disc_data_col_i, const arma::Col<int>& disc_data_col_j);

double get_marginal_ml_entropy(const arma::vec& p_marg);
double get_joint_ml_entropy(const arma::mat& p_joint);
int get_n_gene_pairs(int n_genes);
arma::Mat<int> get_idx_lookup_mat(int n_genes);
std::vector<std::pair<int, int>> get_ij_list(int n_genes);

#endif
