import os
import sys
from timeit import timeit

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

# from mpi4py import MPI

sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
from distance import knn, get_rank #euc_distance, cos_sim

seed = 42
# Length of array set
n_len = 1000
# Number of columns in an arbitrary dataset
dims = 256
# Similarity formula
met = 'cosine'

rng = np.random.default_rng(seed)
X = rng.random((n_len, dims))
Y = rng.random((n_len, dims))
k = 15

def compute_with_python(metric=met, x=X, y=Y):
    """
    Calculate a naive version of euclidean distance for two arrays.
    """
    if metric == 'euclidean':
        # shape of num samples in both X and Y
        D = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                for k in range(x.shape[1]):
                    diff_sq = (x[i][k] - y[j][k])**2
                D[i][j] += np.sqrt(diff_sq)

    elif metric == 'cosine':
        A = np.zeros((x.shape[0], y.shape[0]))
        D = np.zeros((x.shape[0], y.shape[0]))

        # Create A
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                for k in range(x.shape[1]):
                    A[i][j] += x[i][k] * y[j][k]
        
        Xnorm = np.sqrt(np.sum(x * x, axis=1))
        Ynorm = np.sqrt(np.sum(y * y, axis=1))        

        # Calculate denominator and D
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                denom = (Xnorm[i] * Ynorm[j])
                D[i][j] = 0 if denom == 0 else A[i][j] / denom
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Expected 'cosine' or 'euclidean'.")

    return(D)

def skKnn(x=X, y=Y, metric=met):
    """
    Create the classifier and fit in one call.
    """
    if metric == 'cosine':
        X_norm = normalize(x, norm="l2", axis=1)
        Y_norm = normalize(y, norm="l2", axis=1)
    else:
        X_norm = x
        Y_norm = y

    nneighbors = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        algorithm="auto"
    )
    nneighbors.fit(Y_norm)
    return(nneighbors.kneighbors(X_norm, n_neighbors=k))

if get_rank() == 0:
    print(f"Metric used: ", met)
    print("Are the results correct?")
    print("knn---------------------")

cpp_scrs, cpp_indxs = knn(X, Y, k, False)

if get_rank() == 0:
    py_scrs, py_indxs = skKnn()
    py_scrs = 1 - (py_scrs**2) / 2 # converts norm'd distance to cosine similiarity
    print(f"Score Matrices: {np.allclose(cpp_scrs, py_scrs)}")
    print(f"Index Matrices: {np.array_equal(cpp_indxs, py_indxs)}")

#Check the knn functions
# Set up timing for scikit benchmark and C++
# Vary the number of calculations based on the time it takes to run.
# num_evaluations = int(10 / timeit(lambda: knn(X, Y, k, False), number=1))
# num_evaluations = max(1, num_evaluations)
num_evaluations = 10 # must be fixed for MPI

# Measure naive computation in seconds
# Measure scikit learn's computation in seconds.
if get_rank() == 0:
    sci_time = timeit(lambda: skKnn(), number=num_evaluations)
# Measure distance's computation in seconds.
c_time = timeit(lambda: knn(X, Y, k, False), number=num_evaluations)
#Avg the total time over number of repeat calculations.
if get_rank() == 0:
    print("[Scikit] Time per evaluation: {:10.0f} microseconds.".format(1e6 * sci_time / num_evaluations))
    print("[C++] Time per evaluation: {:10.0f} microseconds.".format(1e6 * c_time / num_evaluations))
