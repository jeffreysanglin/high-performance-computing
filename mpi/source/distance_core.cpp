#include <mpi.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <limits>
#include <vector>
#include <cblas.h>

void mpi_initializer() {
    int initialized = 0;
    int finalized = 0;

    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);

    if (finalized) {
        throw std::runtime_error("MPI was already finalized");
    }

    if (!initialized) {
        int provided = 0;

        MPI_Init_thread(
            nullptr,
            nullptr,
            MPI_THREAD_FUNNELED,
            &provided
        );

        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI implementation does not support MPI_THREAD_FUNNELED");
        }
    }
}

void mpi_finalize() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

// MPI KNN
void knn(
	const double* X_in,
	const double* Y_in,
    double* Scores_out,
    int* Indices_out,
    const int n_rows,
    const int n_dims,
	const int n_neighbors,
	const bool exclude_self
) {
    // ========================================
	// MPI Setup
	// ========================================

    mpi_initializer();

    //Get MPI Task Rank
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Get MPI Size
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	// ========================================
	// Cosine similarity computation
	// ========================================

	// shape[0] == rows
	ssize_t Xrows = n_rows;
	ssize_t Xfeat = n_dims;
	ssize_t Yrows = n_rows;
	ssize_t Yfeat = n_dims;

    // Prep some admin work to send local X's to ranks
    int base = Xrows / size;
    int rem = Xrows % size;

    // Find the first row for each rank.
    int my_first_row = rank * base + std::min(rank, rem);
    // Distribute the remainder evenly among the first ranks within "remainder range"
    int my_row_count = base + (rank < rem ? 1 : 0);
    // Find the last row for each rank, assuming exclusive ceiling.
    int my_last_row = my_first_row + my_row_count; 

    // my_first is the starting row (int); multiply by features to get the vectorized position
    const double* local_X = X_in + my_first_row * Xfeat;

	//Create intermediary arrays to handle numerator and denominator
    std::vector<double> local_A(my_row_count * Yrows);

	//Numerator = Matrix Multiplication Xi * Yj
	//Dot product using BLAS: X * Y^T
	if (Xfeat != Yfeat) {
		throw std::runtime_error("Feature dimensions must match");
	}
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		my_row_count,
		Yrows,
		Xfeat,
		1.0,
		local_X,
		Xfeat,
		Y_in,
		Yfeat,
		0.0,
		local_A.data(),
		Yrows
	);

	//Norm Vectors, ||xi|| (or nx[i]) and ||yj|| (or ny[j])
	//which is... sqrt( sum_k X[i,k]^2)
	//Xnorm - each rank computes separately
    std::vector<double> local_Xnorm(my_row_count);
    #pragma omp parallel for
	for (int i = 0; i < my_row_count; ++i) {
		local_Xnorm[i] = cblas_dnrm2(Xfeat, local_X + i * Xfeat, 1);
	}

	//Ynorm - rank 0 computes and broadcasts
    std::vector<double> Ynorm(Yrows, 0.0);
    if (rank == 0) {  
        #pragma omp parallel for
        for (int j = 0; j < Yrows; ++j) {
            Ynorm[j] = cblas_dnrm2(Yfeat, Y_in + j * Yfeat, 1);
        }
    
    }

    MPI_Bcast(Ynorm.data(), Yrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//Denominator = outer product of nx[i] * ny[j], ie nx * ny^T
    std::vector<double> local_D(my_row_count * Yrows);
    
    #pragma omp parallel for
	for (int i = 0; i < my_row_count; ++i) {
		for (int j = 0; j < Yrows; ++j) {
			double denom = (local_Xnorm[i] * Ynorm[j]);
            int idx = i * Yrows + j;
			// If not zero, calculate every combo of i and j and divide by A.
			local_D[idx] = denom == 0.0 ? 0.0 : local_A[idx] / denom;
		}
	}

    // Hack to run the whole code successfully.
    //MPI_Bcast(D, Xrows * Yrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //OUTPUT: D is our cosine similarity.

	// ========================================
	// End cosine similarity computation
	// ========================================


	// ========================================
	// KNN indices and scores computation
	// ========================================
    std::vector<double> local_Scrs(my_row_count * n_neighbors);
    std::vector<int> local_Idxs(my_row_count * n_neighbors);

	// loop through i, and find the largest k values looping through j.
	//naive looping
	// if (rank == 0) {
    // #pragma omp parallel for schedule(static)
    for (int local_i = 0; local_i < my_row_count; ++local_i) {
        int global_i = my_first_row + local_i; //where rank starts, plus the current i = the global i element.
        //create a vector to know when to skip an element (diagonal or prior used max)
        // std::vector<bool> skip(Dcols, false);
        std::vector<int> idx(Yrows);

        // assign the index int for every column.
        for (int j = 0; j < Yrows; ++j) {
            idx[j] = j;
        }

        // remove the element idx + i if diagonal and excluding self.
        if (exclude_self && Xrows == Yrows) {
            idx.erase(idx.begin() + global_i);
        }

        int actual_k = std::min<int>(n_neighbors, idx.size());

        int i_yrows = local_i * Yrows;
        // sort for the top k elements
        std::partial_sort(
            idx.begin(),                // the start of the range
            idx.begin() + actual_k,  // the middle of the range
            idx.end(),                  // the end of the range
            [&](int a, int b) {
                return local_D[i_yrows + a] > local_D[i_yrows + b]; // does index a come before b?
            }
        );

        int i_n = local_i * n_neighbors;
        for (int k = 0; k < actual_k; ++k) {
            int j = idx[k];
            int ink = i_n + k;
            local_Scrs[ink] = local_D[i_yrows + j];
            local_Idxs[ink] = j;
        }
    }

    // Get stuff ready to gather back -- create receive counts and displacements for rnk 0
    std::vector<int> root_recv_counts;
    std::vector<int> root_displacements;

    if (rank == 0) {
        root_recv_counts.resize(size);
        root_displacements.resize(size);

        for (int r = 0; r < size; ++r) {
            int r_row_count = base + (r < rem ? 1 : 0);
            int r_first_row = r * base + std::min(r, rem);

            root_recv_counts[r] = r_row_count * n_neighbors;
            root_displacements[r] = r_first_row * n_neighbors;
        }
    }

    //Gather all the scores
    MPI_Gatherv(
        local_Scrs.data(),                //pointer to the buffer
        my_row_count * n_neighbors,   //number of elements to send
        MPI_DOUBLE,             //datatype to send
        Scores_out,                      //buffer to receive in
        rank == 0 ? root_recv_counts.data() : nullptr,   //count to receive
        rank == 0 ? root_displacements.data() : nullptr,
        MPI_DOUBLE,             //data type to receive
        0,                      //root rank
        MPI_COMM_WORLD
    );

    //Gather all the indices
    MPI_Gatherv(
        local_Idxs.data(),                //pointer to the buffer
        my_row_count * n_neighbors,   //number of elements to send
        MPI_INT,             //datatype to send
        Indices_out,                      //buffer to receive in
        rank == 0 ? root_recv_counts.data() : nullptr,   //count to receive
        rank == 0 ? root_displacements.data() : nullptr,
        MPI_INT,             //data type to receive
        0,                      //root rank
        MPI_COMM_WORLD
    );
    // }

    // ========================================
    // End knn indices and scores computation
    // ========================================
}