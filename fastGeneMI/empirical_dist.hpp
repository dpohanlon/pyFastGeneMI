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
//  These functions get empirical probability distributions from discretised data
// ----------------------------------------------------------------------------------
#include <armadillo>
#include <limits>
#include <stdexcept>
#include <string>

inline void validate_disc_column(const arma::Col<int>& disc_data, const std::string& name)
{
  if(disc_data.n_elem == 0)
  {
    throw std::invalid_argument(name + " must be non-empty");
  }

  if(disc_data.min() < 0)
  {
    throw std::invalid_argument(name + " contains negative values; discrete labels must be >= 0");
  }
}

inline arma::uword get_n_bins_from_column(const arma::Col<int>& disc_data, const std::string& name)
{
  validate_disc_column(disc_data, name);
  return static_cast<arma::uword>(disc_data.max()) + 1;
}

inline arma::vec get_emp_marg_dist(const arma::Col<int>& disc_data_col)
{
  validate_disc_column(disc_data_col, "disc_data_col");

  const arma::uword n_samples = disc_data_col.n_elem;
  const arma::uword n_bins = get_n_bins_from_column(disc_data_col, "disc_data_col");

  arma::vec counts(n_bins, arma::fill::zeros);

  for(arma::uword k = 0; k < n_samples; ++k)
  {
    const arma::uword idx = static_cast<arma::uword>(disc_data_col(k));
    counts(idx) += 1.0;
  }

  return counts / static_cast<double>(n_samples);
}

inline arma::mat get_emp_joint_dist(const arma::Col<int>& disc_data_col_i, const arma::Col<int>& disc_data_col_j)
{
  if(disc_data_col_i.n_elem != disc_data_col_j.n_elem)
  {
    throw std::invalid_argument("disc_data_col_i and disc_data_col_j must have the same length");
  }

  validate_disc_column(disc_data_col_i, "disc_data_col_i");
  validate_disc_column(disc_data_col_j, "disc_data_col_j");

  const arma::uword n_samples = disc_data_col_i.n_elem;
  const arma::uword n_bins_i = get_n_bins_from_column(disc_data_col_i, "disc_data_col_i");
  const arma::uword n_bins_j = get_n_bins_from_column(disc_data_col_j, "disc_data_col_j");

  const std::size_t bins_i = static_cast<std::size_t>(n_bins_i);
  const std::size_t bins_j = static_cast<std::size_t>(n_bins_j);

  if(bins_i != 0 && bins_j > std::numeric_limits<std::size_t>::max() / bins_i)
  {
    throw std::overflow_error("joint histogram dimensions overflow");
  }

  arma::mat counts2d(n_bins_i, n_bins_j, arma::fill::zeros);

  for(arma::uword k = 0; k < n_samples; ++k)
  {
    const arma::uword idx_i = static_cast<arma::uword>(disc_data_col_i(k));
    const arma::uword idx_j = static_cast<arma::uword>(disc_data_col_j(k));
    counts2d(idx_i, idx_j) += 1.0;
  }

  return counts2d / static_cast<double>(n_samples);
}
