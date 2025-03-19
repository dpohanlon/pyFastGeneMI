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
//  Utility functions
// ----------------------------------------------------------------------------------

#include "fastGeneMI.h"

// ----------------------------------------------------------------------------------
//  Maximum Likelihood Entropy functions
// ----------------------------------------------------------------------------------

// Maximum Likelihood marginal entropy from marginal probability
double get_marginal_ml_entropy(const arma::vec& p_marg)
{
  return -arma::sum(p_marg % arma::log(p_marg + 1e-16));
}

// Maximum likelihood joint entropy from joint probability
double get_joint_ml_entropy(const arma::mat& p_joint)
{
  return -arma::accu(p_joint % arma::log(p_joint + 1e-16));
}

// ----------------------------------------------------------------------------------
//  Build a matrix whose entries are the index of flattened 1d array of
//  the upper-triangle of a n_genes x n_genes matrix
// ----------------------------------------------------------------------------------

int get_n_gene_pairs(const int n_genes)
{
    return static_cast<int>( (static_cast<double>(n_genes)/2.0) * static_cast<double>(n_genes+1) );
}

arma::Mat<int> get_idx_lookup_mat(const int n_genes)
{
  arma::Mat<int> idx_lookup(n_genes, n_genes, arma::fill::zeros);
  int ij = 0;
  for(int i = 0; i < n_genes; ++i)
  {
    for(int j = i; j < n_genes; ++j)
    {
      idx_lookup(i, j) = ij;
      ++ij;
    }
  }
  return idx_lookup;
}

std::vector<std::pair<int,int> > get_ij_list(const int n_genes)
{
  const int n_pairs = get_n_gene_pairs(n_genes);
  std::vector<std::pair<int,int> > ij_list(n_pairs);
  int ij = 0;

  for(int i = 0; i < n_genes; ++i)
  {
    for(int j = i; j < n_genes; ++j)
    {
      ij_list[ij] = std::make_pair(i, j);
      ++ij;
    }
  }

  return ij_list;
}
