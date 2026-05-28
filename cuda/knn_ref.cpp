// knn_ref.hpp
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cblas.h>

constexpr int K = 15;

void knn_serial(
    const std::vector<float>& X,
    const std::vector<float>& Y,
    std::vector<float>& scores,
    std::vector<int>& indices,
    int rows,
    int dims
) {
    // Inputs for D
    std::vector<float> D(rows * rows);
    std::vector<float> A(rows * rows);

    // Cosine simliarity
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans, 
        CblasTrans,
        static_cast<int>(rows),
        static_cast<int>(rows),
        static_cast<int>(dims),
        1.0,
        X.data(),
        static_cast<int>(dims),
        Y.data(),
        static_cast<int>(dims),
        0.0,
        A.data(),
        static_cast<int>(rows)
    );

    std::vector<float> Xnorm(rows, 0.0);
    std::vector<float> Ynorm(rows, 0.0);
    for (int i = 0; i < rows; ++i) {
        Xnorm[i] = cblas_snrm2(dims, X.data() + i * dims, 1);
    }
    for (int j = 0; j < rows; ++j) {
        Ynorm[j] = cblas_snrm2(dims, Y.data() + j * dims, 1);
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < rows; ++j) {
            float denom = Xnorm[i] * Ynorm[j];
            D[i * rows + j] = denom == 0.0
                ? 0.0
                : A[i * rows + j] / denom;
        }
    }

    //KNN
    // loop through i, and find the largest k values looping through j.
    //naive looping
    for (int i = 0; i < rows; ++i) {
        //create a vector to know when to skip an element (diagonal or prior used max)
        // std::vector<bool> skip(Dcols, false);
        std::vector<int> idx(rows);
        
        // assign the index int for every column.
        for (int j = 0; j < rows; ++j) {
            idx[j] = j;
        }

        
        // sort for the top k elements
        std::partial_sort(
            idx.begin(),                // the start of the range
            idx.begin() + K,  // the middle of the range
            idx.end(),                  // the end of the range
            [&](int a, int b) {
                return D[i * rows + a] > D[i * rows + b]; // does index a come before b?
            }
        );

        for (int k = 0; k < K; ++k) {
            int j = idx[k];
            scores[i * K + k] = D[i * rows + j];
            indices[i * K + k] = j;
        }
    }
}