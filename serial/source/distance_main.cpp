#include <cmath>
#include <vector>
#include <random>
#include "distance_core.cpp"

const int K = 15;
const int SEED = 42;
const bool EXCLUDE_SELF = false;

int main(int argc, char** argv) {

    //Create rand X and Y
    int rows = 1000;
    int dims = 256;
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
    cos_sim(X.data(), Y.data(), D.data(), rows, dims);
    knn(D.data(), K, Scores.data(), Indices.data(), K, EXCLUDE_SELF);

    return 0;
}