#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <vector>

#include <cblas.h>

void euc_distance(
    const double* X_in,
    const double* Y_in,
    double* D_out,
    const int n_rows,
    const int n_dims) 
{

    // shape[0] == rows
    ssize_t Xrows = n_rows;
    ssize_t Xfeat = n_dims;
    ssize_t Yrows = n_rows;
    ssize_t Yfeat = n_dims;

    // Create vectors to contain our norms from refactoring (x - y)^2 below.
    std::vector<double> X_norm(Xrows, 0.0);
    std::vector<double> Y_norm(Yrows, 0.0);
    
    //Calc row norms for X
    // #pragma omp parallel for schedule(static)
    for (ssize_t i = 0; i < Xrows; ++i) {
        const double* xrow = X_in + i * Xfeat;
        double sum = 0.0;
        for (ssize_t k = 0; k < Xfeat; ++k) {
            double val = xrow[k];
            sum += val * val;
        }
        X_norm[i] = sum;
    }

    //Calc row norms for Y
    // #pragma omp parallel for schedule(static)
    for (ssize_t j = 0; j < Yrows; ++j) {
        const double* yrow = Y_in + j * Yfeat;
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
        X_in,
        static_cast<int>(Xfeat),
        Y_in,
        static_cast<int>(Yfeat),
        0.0,
        D_out,
        static_cast<int>(Yrows)
    );

    // #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < Xrows; ++i) {
        for (ssize_t j = 0; j < Yrows; ++j) {
            double val = X_norm[i] + Y_norm[j] - 2 * D_out[i * Yrows + j];
            
            // Protect against tiny negative values from rounding
            if (val < 0.0 && val > -1e-12) {
                val = 0.0;
            }

            D_out[i * Yrows + j] = std::sqrt(val);
        }
    }
}

void cos_sim(
    const double* X_in,
    const double* Y_in,
    double* D_out,
    const int n_rows,
    const int n_dims) 
{
    // shape[0] == rows
    ssize_t Xrows = n_rows;
    ssize_t Xfeat = n_dims;
    ssize_t Yrows = n_rows;;
    ssize_t Yfeat = n_dims;

    std::vector<double> A(n_rows * n_rows, 0.0);

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
        X_in,
        static_cast<int>(Xfeat),
        Y_in,
        static_cast<int>(Yfeat),
        0.0,
        A.data(),
        static_cast<int>(Yrows)
    );

    //Norm Vectors, ||xi|| (or nx[i]) and ||yj|| (or ny[j])
    //which is... sqrt( sum_k X[i,k]^2)
    std::vector<double> Xnorm(Xrows, 0.0);
    std::vector<double> Ynorm(Yrows, 0.0);

    //Xnorm
    // #pragma omp parallel for
    for (int i = 0; i < Xrows; ++i) {
        Xnorm[i] = cblas_dnrm2(Xfeat, X_in + i * Xfeat, 1);
    }
    //Ynorm
    // #pragma omp parallel for
    for (int j = 0; j < Yrows; ++j) {
        Ynorm[j] = cblas_dnrm2(Yfeat, Y_in + j * Yfeat, 1);
    }
    

    //Denominator = outer product of nx[i] * ny[j], ie nx * ny^T
    // #pragma omp parallel for
    for (int i = 0; i < Xrows; ++i) {
        for (int j = 0; j < Yrows; ++j) {
            double denom = (Xnorm[i] * Ynorm[j]);
            // If not zero, calculate every combo of i and j and divide by A.
            D_out[i * Yrows + j] = denom == 0.0 ? 0.0 : A[i * Yrows + j] / denom;
        }
    }
}

void knn(
    const double* D_in,
    const int n_rows,
    double* Scores_out,
    int* Indices_out,
    const int n_neighbors,
    const bool exclude_self
) 
{
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
    const int Drows = n_rows;
    const int Dcols = n_rows;

    // loop through i, and find the largest k values looping through j.
    //naive looping
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < Drows; ++i) {
        //create a vector to know when to skip an element (diagonal or prior used max)
        // std::vector<bool> skip(Dcols, false);
        std::vector<int> idx(Dcols);
        
        // assign the index int for every column.
        for (int j = 0; j < Dcols; ++j) {
            idx[j] = j;
        }

        // remove the element idx + i if diagonal and excluding self.
        if (exclude_self  && Drows == Dcols) {
            idx.erase(idx.begin() + i);
        }
        
        // sort for the top k elements
        std::partial_sort(
            idx.begin(),                // the start of the range
            idx.begin() + n_neighbors,  // the middle of the range
            idx.end(),                  // the end of the range
            [&](int a, int b) {
                return D_in[i * Dcols + a] > D_in[i * Dcols + b]; // does index a come before b?
            }
        );

        for (int k = 0; k < n_neighbors; ++k) {
            int j = idx[k];
            Scores_out[i * n_neighbors + k] = D_in[i * Dcols + j];
            Indices_out[i * n_neighbors + k] = j;
        }
    }
}