#include <armadillo>

#include "mi_estimators.hpp"

// Main function for standalone execution
int main() {
  // Example: Create a sample matrix.
  // The sample matrix is assumed to be in R indexing (1-indexed), so the code subtracts 1.
  arma::mat sampleData = {
    {1, 2, 3},
    {3, 2, 1},
    {2, 3, 1},
    {1, 1, 2}
  };

  int n_cores = 2;  // Set desired number of threads

  // Example usage: Compute mutual information using Maximum Likelihood estimator.
  arma::mat miMatrix = mim_ML_cpp(sampleData, n_cores);

  // Print the result to the console
  miMatrix.print("Mutual Information Matrix (ML estimator):");

  return 0;
}
