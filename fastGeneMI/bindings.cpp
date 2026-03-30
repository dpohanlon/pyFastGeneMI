#include <carma>
#include <climits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

#include "mi_estimators.hpp"

namespace py = pybind11;

arma::Mat<int> py_array_to_int_mat(const py::array& input)
{
  if(input.ndim() != 2)
  {
    throw py::value_error("disc_expr_data must be a 2D NumPy array");
  }

  const std::string kind = py::str(input.dtype().attr("kind"));
  if(kind != "i" && kind != "u")
  {
    throw py::type_error("disc_expr_data must have an integer dtype");
  }

  py::array_t<long long, py::array::c_style | py::array::forcecast> arr(input);
  auto view = arr.unchecked<2>();

  arma::Mat<int> out(view.shape(0), view.shape(1));

  for(py::ssize_t i = 0; i < view.shape(0); ++i)
  {
    for(py::ssize_t j = 0; j < view.shape(1); ++j)
    {
      const long long value = view(i, j);

      if(value < 0)
      {
        throw py::value_error("disc_expr_data contains negative values; discrete labels must be >= 0");
      }

      if(value > static_cast<long long>(INT_MAX))
      {
        throw py::value_error("disc_expr_data contains values too large for 32-bit int storage");
      }

      out(static_cast<arma::uword>(i), static_cast<arma::uword>(j)) = static_cast<int>(value);
    }
  }

  return out;
}

py::array_t<double> mim_ML_py(const py::array& disc_expr_data, int n_cores)
{
  arma::Mat<int> data = py_array_to_int_mat(disc_expr_data);
  return carma::mat_to_arr(mim_ML_cpp(data, n_cores));
}

py::array_t<double> mim_MM_py(const py::array& disc_expr_data, int n_cores)
{
  arma::Mat<int> data = py_array_to_int_mat(disc_expr_data);
  return carma::mat_to_arr(mim_MM_cpp(data, n_cores));
}

py::array_t<double> mim_CS_py(const py::array& disc_expr_data, int n_cores)
{
  arma::Mat<int> data = py_array_to_int_mat(disc_expr_data);
  return carma::mat_to_arr(mim_CS_cpp(data, n_cores));
}

PYBIND11_MODULE(_fastGeneMI, m)
{
  m.doc() = "pybind11 bindings for fastGeneMI mutual information estimators";

  m.def("mim_ML_cpp", &mim_ML_py, py::arg("disc_expr_data"), py::arg("n_cores") = 1);
  m.def("mim_MM_cpp", &mim_MM_py, py::arg("disc_expr_data"), py::arg("n_cores") = 1);
  m.def("mim_CS_cpp", &mim_CS_py, py::arg("disc_expr_data"), py::arg("n_cores") = 1);
}
