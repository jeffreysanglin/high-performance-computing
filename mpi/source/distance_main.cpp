#include <mpi.h>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "distance_core.cpp"

const int K = 15;
const int SEED = 42;
const bool EXCLUDE_SELF = false;

int main(int argc, char** argv) {
    
    //Create rand X and Y
    int rows = 10000;
    int dims = 512;
    std::vector<double> X(rows * dims);
    std::vector<double> Y(rows * dims);
    std::vector<double> D(rows * rows);

    //fill the X, Y
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (auto& v : X) v = dist(rng);
    for (auto& v : Y) v = dist(rng);

    //Create Scores, Indices
    std::vector<double> Scores(rows * K);
    std::vector<int> Indices(rows * K);

    //Call the functions
    knn(X.data(), Y.data(), Scores.data(), Indices.data(), rows, dims, K, EXCLUDE_SELF);

    mpi_finalize();

    return 0;
}