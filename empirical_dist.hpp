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

// Return empirical marginal distributions of a discretised dataset
arma::vec get_emp_marg_dist(const arma::Mat<int>& disc_data_col)
{
  const int n_samples(disc_data_col.n_rows);

  const int n_bins = disc_data_col.max() + 1;
  arma::vec p_marginal(n_bins);

  // Loop through each bin
  for(int k = 0; k < n_bins; ++k)
  {
    arma::umat s = arma::sum(disc_data_col == k);
    arma::mat s2 = arma::conv_to<arma::mat>::from(s);
    p_marginal(k) = s2(0) / static_cast<double>(n_samples);
  }
  return p_marginal;
}

// Return the empirical joint distributions of a discrete dataset
arma::mat get_emp_joint_dist(const arma::Mat<int>& disc_data_col_i, const arma::Mat<int>& disc_data_col_j)
{
  const int n_samples(disc_data_col_i.n_rows);

  // This requires every bin to have at least one entry
  const int n_bins_i = disc_data_col_i.max() + 1;
  const int n_bins_j = disc_data_col_j.max() + 1;

  arma::mat counts2d(n_bins_i, n_bins_j, arma::fill::zeros);
  for(int k = 0; k < n_samples; ++k)
  {
    counts2d(disc_data_col_i(k), disc_data_col_j(k)) += 1.0;
  }

  return counts2d / static_cast<double>(n_samples);
}
