#include <mpi.h>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <limits>
#include <vector>
#include <cblas.h>
#include "distance_core.cpp"

namespace py = pybind11;

int get_rank() {
    mpi_initializer();
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

// MPI KNN
pybind11::tuple knn_py(
	const pybind11::array_t<double>& X_in,
	const pybind11::array_t<double>& Y_in,
	const int n_neighbors = 15,
	const bool exclude_self = false
) {
	// ========================================
	// Cosine similarity computation
	// ========================================
	// Get the metadata from our python arrays
	auto X_buf = X_in.request();
	auto Y_buf = Y_in.request();

	// shape[0] == rows
	ssize_t Xrows = X_buf.shape[0];
	ssize_t Xfeat = X_buf.shape[1];

	// Get our local arrays ready for each rank.
	const double* X = static_cast<const double*>(X_buf.ptr);
	const double* Y = static_cast<const double*>(Y_buf.ptr);

    py::array_t<double> scores({static_cast<py::ssize_t>(Xrows),
                            static_cast<py::ssize_t>(n_neighbors)});

    py::array_t<int> indices({static_cast<py::ssize_t>(Xrows),
                         static_cast<py::ssize_t>(n_neighbors)});
    auto score_bfr = scores.request();
    auto index_bfr = indices.request();
    double* Scrs = static_cast<double*>(score_bfr.ptr);
    int* Idxs = static_cast<int*>(index_bfr.ptr);

	knn(
        X,
        Y,
        Scrs,
        Idxs,
        Xrows,
        Xfeat,
        n_neighbors,
        exclude_self
    );
    
    int rank = get_rank();

    if (rank == 0) {
        return py::make_tuple(scores, indices);
    }
    return py::make_tuple(
        py::array_t<double>(0),
        py::array_t<int>(0)
    );
}

PYBIND11_MODULE(distance_py, m)
{
	m.def("knn_py", &knn_py);
    m.def("get_rank", &get_rank);
}