#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <limits>
#include <vector>

#include "distance_core.cpp"

namespace py = pybind11;

pybind11::array_t<double> euc_distance_py(const pybind11::array_t<double> &X_in,
        const pybind11::array_t<double> &Y_in) {
    // Get the metadata from our python arrays
    auto X_buf = X_in.request();
    auto Y_buf = Y_in.request();

    // shape[0] == rows
    ssize_t Xrows = X_buf.shape[0];
    ssize_t Xfeat = X_buf.shape[1];
    ssize_t Yrows = Y_buf.shape[0];
    ssize_t Yfeat = Y_buf.shape[1];

    /* 
        Cast raw buffer pointer to C++ double pointer. 
        Now we can access the arrays as if flat C array
    */
    const double* X = static_cast<const double*>(X_buf.ptr);
    const double* Y = static_cast<const double*>(Y_buf.ptr);

    // Create an output array that py can handle using array_t
    py::array_t<double> D_out({Xrows, Yrows});
    auto D_buf = D_out.request();
    double* D = static_cast<double*>(D_buf.ptr);

    euc_distance(X, Y, D, Xrows, Xfeat);
    return D_out;
}


pybind11::array_t<double> cos_sim_py(
    const pybind11::array_t<double>& X_in,
    const pybind11::array_t<double>& Y_in
) {
    // Get the metadata from our python arrays
    auto X_buf = X_in.request();
    auto Y_buf = Y_in.request();

    // shape[0] == rows
    ssize_t Xrows = X_buf.shape[0];
    ssize_t Xfeat = X_buf.shape[1];
    ssize_t Yrows = Y_buf.shape[0];
    ssize_t Yfeat = Y_buf.shape[1];

    /* 
        Cast raw buffer pointer to C++ double pointer. 
        Now we can access the arrays as if flat C array
    */
    const double* X = static_cast<const double*>(X_buf.ptr);
    const double* Y = static_cast<const double*>(Y_buf.ptr);

    // Create an output array that py can handle using array_t
    py::array_t<double> D_out({Xrows, Yrows});
    auto D_buf = D_out.request();
    double* D = static_cast<double*>(D_buf.ptr);

    cos_sim(X, Y, D, Xrows, Xfeat);
    return D_out;
}

pybind11::tuple knn_py(
    const pybind11::array_t<double>& D_in,
    const int n_neighbors = 15,
    const bool exclude_self = false
) {
    /*
    Rule of thumb used
        n≤1,000:
        → k=5 to 15
        1,000≤n≤50,000:
        → k=10 to 30
        n≥50,000:
        → k=15 to 50+
    */
    // Inputs for D
    auto D_buf = D_in.request();
    const int Drows = D_buf.shape[0];
    const int Dcols = D_buf.shape[1];
    const double* D = static_cast<const double*>(D_buf.ptr);

    // Create output array for k indices
    py::array_t<int> indices({Drows, n_neighbors});
    auto ind_buf = indices.request();
    int* Idxs = static_cast<int*>(ind_buf.ptr);
    // Create array for scores
    py::array_t<double> scores({Drows, n_neighbors});
    auto score_buf = scores.request();
    double* Scrs = static_cast<double*>(score_buf.ptr);

    knn(D, Drows, Scrs, Idxs, n_neighbors, exclude_self);
    return py::make_tuple(scores, indices);
}

PYBIND11_MODULE(distance_py, m)
{
    m.def("euc_distance_py", &euc_distance_py);
    m.def("cos_sim_py", &cos_sim_py);
    m.def("knn_py", &knn_py);
}