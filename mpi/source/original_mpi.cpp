#include <mpi.h>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <limits>
#include <vector>

#include <cblas.h>

namespace py = pybind11;

pybind11::array_t<double> euc_distance(const pybind11::array_t<double> &X_in,
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

    // Create vectors to contain our norms from refactoring (x - y)^2 below.
    std::vector<double> X_norm(Xrows, 0.0);
    std::vector<double> Y_norm(Yrows, 0.0);
    
    //Calc row norms for X
    #pragma omp parallel for schedule(static)
    for (ssize_t i = 0; i < Xrows; ++i) {
        const double* xrow = X + i * Xfeat;
        double sum = 0.0;
        for (ssize_t k = 0; k < Xfeat; ++k) {
            double val = xrow[k];
            sum += val * val;
        }
        X_norm[i] = sum;
    }

    //Calc row norms for X
    #pragma omp parallel for schedule(static)
    for (ssize_t j = 0; j < Yrows; ++j) {
        const double* yrow = Y + j * Yfeat;
        double sum = 0.0;
        for (ssize_t k = 0; k < Xfeat; ++k) {
            double val = yrow[k];
            sum += val * val;
        }
        Y_norm[j] = sum;
    }

    //Dot product using BLAS: X * Y^T
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans, 
        CblasTrans,
        static_cast<int>(Xrows),
        static_cast<int>(Yrows),
        static_cast<int>(Xfeat),
        1.0,
        X,
        static_cast<int>(Xfeat),
        Y,
        static_cast<int>(Yfeat),
        0.0,
        D,
        static_cast<int>(Yrows)
    );

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < Xrows; ++i) {
        for (ssize_t j = 0; j < Yrows; ++j) {
            double val = X_norm[i] + Y_norm[j] - 2 * D[i * Yrows + j];
            
            // Protect against tiny negative values from rounding
            if (val < 0.0 && val > -1e-12) {
                val = 0.0;
            }

            D[i * Yrows + j] = std::sqrt(val);
        }
    }
    return D_out;
}


pybind11::array_t<double> cos_sim(
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

    //Create intermediary arrays to handle numerator and denominator
    py::array_t<double> A_out({Xrows, Yrows});
    auto A_buf = A_out.request();
    double* A = static_cast<double*>(A_buf.ptr);

    // Create an output array that py can handle using array_t
    py::array_t<double> D_out({Xrows, Yrows});
    auto D_buf = D_out.request();
    double* D = static_cast<double*>(D_buf.ptr);

    //Numerator = Matrix Multiplication Xi * Yj
    //Dot product using BLAS: X * Y^T
    if (Xfeat != Yfeat) {
       throw std::runtime_error("Feature dimensions must match");
    }

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans, 
        CblasTrans,
        static_cast<int>(Xrows),
        static_cast<int>(Yrows),
        static_cast<int>(Xfeat),
        1.0,
        X,
        static_cast<int>(Xfeat),
        Y,
        static_cast<int>(Yfeat),
        0.0,
        A,
        static_cast<int>(Yrows)
    );

    //Norm Vectors, ||xi|| (or nx[i]) and ||yj|| (or ny[j])
    //which is... sqrt( sum_k X[i,k]^2)
    std::vector<double> Xnorm(Xrows, 0.0);
    std::vector<double> Ynorm(Yrows, 0.0);

    //Xnorm
    for (int i = 0; i < Xrows; ++i) {
        Xnorm[i] = cblas_dnrm2(Xfeat, X + i * Xfeat, 1);
    }
    //Ynorm
    for (int j = 0; j < Yrows; ++j) {
        Ynorm[j] = cblas_dnrm2(Yfeat, Y + j * Yfeat, 1);
    }
    

    //Denominator = outer product of nx[i] * ny[j], ie nx * ny^T
    for (int i = 0; i < Xrows; ++i) {
        for (int j = 0; j < Yrows; ++j) {
            double denom = (Xnorm[i] * Ynorm[j]);
            // If not zero, calculate every combo of i and j and divide by A.
            D[i * Yrows + j] = denom == 0.0 ? 0.0 : A[i * Yrows + j] / denom;
        }
    }

    return D_out;
}

// MPI KNN
pybind11::tuple knn(
    const pybind11::array_t<double>& X_in,
    const pybind11::array_t<double>& Y_in,
    const int n_neighbors = 15,
    const bool exclude_self = false
) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int Xrows = 0;
    int Xfeat = 0;
    int Yrows = 0;
    int Yfeat = 0;

    std::vector<double> X_vec;
    std::vector<double> Y_vec;

    if (rank == 0) {
        auto X_buf = X_in.request();
        auto Y_buf = Y_in.request();

        Xrows = static_cast<int>(X_buf.shape[0]);
        Xfeat = static_cast<int>(X_buf.shape[1]);
        Yrows = static_cast<int>(Y_buf.shape[0]);
        Yfeat = static_cast<int>(Y_buf.shape[1]);

        if (Xfeat != Yfeat) {
            throw std::runtime_error("Feature dimensions must match");
        }

        const double* X0 = static_cast<const double*>(X_buf.ptr);
        const double* Y0 = static_cast<const double*>(Y_buf.ptr);

        X_vec.assign(X0, X0 + Xrows * Xfeat);
        Y_vec.assign(Y0, Y0 + Yrows * Yfeat);
    }

    MPI_Bcast(&Xrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Xfeat, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Yrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Yfeat, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        X_vec.resize(Xrows * Xfeat);
        Y_vec.resize(Yrows * Yfeat);
    }

    const int base = Xrows / size;
    const int rem = Xrows % size;

    const int local_rows = base + (rank < rem ? 1 : 0);
    const int row_start = rank * base + std::min(rank, rem);

    std::vector<int> scatter_counts(size);
    std::vector<int> scatter_displs(size);

    std::vector<double> local_scores(local_rows * n_neighbors);
    std::vector<int> local_indices(local_rows * n_neighbors);

    // Create separate counts and displs for each rank.
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            int r_rows = base + (r < rem ? 1 : 0); //Earlier rank gets 1 additional row.
            int r_start = r * base + std::min(r, rem);

            scatter_counts[r] = r_rows * Xfeat;
            scatter_displs[r] = r_start * Xfeat;
        }
    }

    std::vector<double> local_X(local_rows * Xfeat);
    
    //Measuring communication - Measure Scatter
    double t0, t1, t_scatter, t_bcast, t_compute, t_gather;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    // Scatter X instead of Bcast
    MPI_Scatterv(
        rank == 0 ? X_vec.data() : nullptr,
        scatter_counts.data(),
        scatter_displs.data(),
        MPI_DOUBLE,
        local_X.data(),
        local_rows * Xfeat,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    //Measure Scatter
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    t_scatter = t1 - t0;
    t0 = MPI_Wtime();

    // MPI Bcast Y
    MPI_Bcast(Y_vec.data(), Yrows * Yfeat, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Measure the Broadcast
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    t_bcast = t1 - t0;

    t0 = MPI_Wtime();

    const double* X = local_X.data();
    const double* Y = Y_vec.data();

    std::vector<double> Ynorm(Yrows);

    // #pragma omp parallel for schedule(static)
    // for (int i = 0; i < Yrows; ++i) {
    //     Xnorm[i] = cblas_dnrm2(Xfeat, local_X + i * Xfeat, 1);
    // }

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < Yrows; ++j) {
        Ynorm[j] = cblas_dnrm2(Yfeat, Y + j * Yfeat, 1);
    }

    #pragma omp parallel for schedule(static)
    for (int li = 0; li < local_rows; ++li) {
        const int i = row_start + li;       //local row start
        const double* x = X + li * Xfeat;   //local row pointer
        const double xnorm = cblas_dnrm2(Xfeat, x, 1);

        std::vector<double> best_score(n_neighbors, -std::numeric_limits<double>::infinity());
        std::vector<int> best_idx(n_neighbors, -1);

        for (int j = 0; j < Yrows; ++j) {
            if (exclude_self && Xrows == Yrows && i == j) {
                continue;
            }

            const double* y = Y + j * Yfeat;
            const double dot = cblas_ddot(Xfeat, x, 1, y, 1);
            const double denom = xnorm * Ynorm[j];
            const double score = denom == 0.0 ? 0.0 : dot / denom;

            int worst = 0;
            for (int k = 1; k < n_neighbors; ++k) {
                if (best_score[k] < best_score[worst]) {
                    worst = k;
                }
            }

            if (score > best_score[worst]) {
                best_score[worst] = score;
                best_idx[worst] = j;
            }
        }

        std::vector<int> order(n_neighbors);
        std::iota(order.begin(), order.end(), 0);

        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return best_score[a] > best_score[b];
        });

        for (int k = 0; k < n_neighbors; ++k) {
            local_scores[li * n_neighbors + k] = best_score[order[k]];
            local_indices[li * n_neighbors + k] = best_idx[order[k]];
        }
    }

    std::vector<int> gather_counts(size);
    std::vector<int> gather_displs(size);

    for (int r = 0; r < size; ++r) {
        const int r_rows = base + (r < rem ? 1 : 0);
        const int r_start = r * base + std::min(r, rem);

        gather_counts[r] = r_rows * n_neighbors;
        gather_displs[r] = r_start * n_neighbors;
    }

    py::array_t<double> scores;
    py::array_t<int> indices;

    double* Scrs = nullptr;
    int* Idxs = nullptr;

    if (rank == 0) {
        scores = py::array_t<double>({Xrows, n_neighbors});
        indices = py::array_t<int>({Xrows, n_neighbors});

        Scrs = static_cast<double*>(scores.request().ptr);
        Idxs = static_cast<int*>(indices.request().ptr);
    }

    //Measure Compute Loop
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    t_compute = t1 - t0;

    t0 = MPI_Wtime();

    MPI_Gatherv(
        local_scores.data(),
        local_rows * n_neighbors,
        MPI_DOUBLE,
        Scrs,
        gather_counts.data(),
        gather_displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gatherv(
        local_indices.data(),
        local_rows * n_neighbors,
        MPI_INT,
        Idxs,
        gather_counts.data(),
        gather_displs.data(),
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Measure Gather
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    t_gather = t1 - t0;

    // Finish measuring and print
    double max_scatter, max_bcast, max_compute, max_gather;

    MPI_Reduce(&t_scatter, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_bcast,   &max_bcast,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_gather,  &max_gather,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "scatter: " << max_scatter << "\n";
        std::cout << "bcast:   " << max_bcast << "\n";
        std::cout << "compute: " << max_compute << "\n";
        std::cout << "gather:  " << max_gather << "\n";
    }

    if (rank == 0) {
        return py::make_tuple(scores, indices);
    }

    return py::make_tuple(py::none(), py::none());
}

PYBIND11_MODULE(distance, m)
{
    m.def("euc_distance", &euc_distance);
    m.def("cos_sim", &cos_sim);
    m.def("knn", &knn);
}