#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include "distance_core_serial.cpp"

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

    // ---- Timing cos_sim ----
    auto t1 = std::chrono::high_resolution_clock::now();

    //Call the functions
    cos_sim(X.data(), Y.data(), D.data(), rows, dims);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::high_resolution_clock::now();
    knn(D.data(), rows, Scores.data(), Indices.data(), K, EXCLUDE_SELF);
    auto t4 = std::chrono::high_resolution_clock::now();

    // Convert to milliseconds
    double cos_time = std::chrono::duration<double, std::micro>(t2 - t1).count();
    double knn_time = std::chrono::duration<double, std::micro>(t4 - t3).count();

    std::cout << "cos_sim time: " << cos_time << " us\n";
    std::cout << "knn time:     " << knn_time << " us\n";
    std::cout << "combined time: " << cos_time + knn_time << " us\n";
    
    return 0;
}