#include <carma>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "mi_estimators.hpp"

namespace py = pybind11;

py::array_t<double> mim_ML_py(const arma::mat& disc_expr_data, int n_cores) {
    arma::mat miMatrix = mim_ML_cpp(disc_expr_data, n_cores);

    return carma::mat_to_arr(miMatrix);
}

py::array_t<double> mim_MM_py(const arma::mat& disc_expr_data, int n_cores) {
    arma::mat miMatrix = mim_MM_cpp(disc_expr_data, n_cores);

    return carma::mat_to_arr(miMatrix);
}

py::array_t<double> mim_CS_py(const arma::mat& disc_expr_data, int n_cores) {
    arma::mat miMatrix = mim_CS_cpp(disc_expr_data, n_cores);

    return carma::mat_to_arr(miMatrix);
}

PYBIND11_MODULE(fastGeneMI, m) {
    m.doc() = "pybind11 bindings for fastGeneMI mutual information estimators using Carma";

    m.def("mim_ML_cpp", &mim_ML_py,
          "Compute mutual information using the maximum likelihood estimator.",
          py::arg("disc_expr_data"), py::arg("n_cores"));

    m.def("mim_MM_cpp", &mim_MM_py,
          "Compute mutual information using the Miller-Maddow estimator.",
          py::arg("disc_expr_data"), py::arg("n_cores"));

    m.def("mim_CS_cpp", &mim_CS_py,
          "Compute mutual information using the Chao-Shen estimator.",
          py::arg("disc_expr_data"), py::arg("n_cores"));
}
