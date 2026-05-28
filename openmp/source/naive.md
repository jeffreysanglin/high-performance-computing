# Naive implementation of knn in cpp

```cpp
pybind11::tuple knn(
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

    // loop through i, and find the largest k values looping through j.
    //naive looping
    for (int i = 0; i < Drows; ++i) {
        //create a vector to know when to skip an element (diagonal or prior used max)
        std::vector<bool> skip(Dcols, false);
        if (exclude_self  && Drows == Dcols) {
            skip[i] = true;
        }
        
        for (int k = 0; k < n_neighbors; ++k) {
            //declare infinitesimally small number.
            double max_k = -std::numeric_limits<double>::infinity();
            int max_idx = -1;

            for (int j = 0; j < Dcols; ++j) {
                if (skip[j]) {
                    continue;
                }
                double val = D[i * Dcols + j];
                if (val > max_k) {
                    max_k = val;
                    max_idx = j;
                }
            }
            Scrs[i * n_neighbors + k] = max_k;
            Idxs[i * n_neighbors + k] = max_idx;
            skip[max_idx] = true;
        }
    }

    return py::make_tuple(scores, indices);
}
```