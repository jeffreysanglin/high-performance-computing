import os
import sys
from timeit import timeit

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sys.path.append(os.path.join(os.path.dirname(__file__), "build"))

seed = 42
# Length of array set
n_len = 1000
if len(sys.argv) > 1:
    n_len = int(sys.argv[1])

# Number of columns in an arbitrary dataset
dims = 256
# Similarity formula
met = 'cosine'

rng = np.random.default_rng(seed)
X = rng.random((n_len, dims))
Y = rng.random((n_len, dims))
k = 15

# sklearn version
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

# mpi4py
def knn(X_in, Y_in, n_neighbors, exclude_self=False):
    """
    """
	# ========================================
	# Cosine similarity computation
	# ========================================
    Xrows = X_in.shape[0]
    Xfeat = X_in.shape[1]
    Yrows = Y_in.shape[0]
    Yfeat = Y_in.shape[1]

    if (Xfeat != Yfeat):
        raise ValueError("Feature dimensions must match")

    if n_neighbors > Yrows:
        raise ValueError("n_neighbors cannot be larger than Y rows")

    # Prep the base/remainder
    base = Xrows // size
    rem = Xrows % size
    my_first_row = rank * base + min(rank, rem)
    my_row_count = base + (1 if rank < rem else 0)
    my_last_row = my_first_row + my_row_count

    if my_row_count == 0:
        local_indices = np.empty((0, n_neighbors), dtype=np.int64)
        local_scores = np.empty((0, n_neighbors), dtype=np.float64)

    else:
        local_X = X_in[my_first_row:my_last_row, :]

        Xnormed = local_X / np.linalg.norm(local_X, axis=1, keepdims=True)
        Ynormed = Y_in / np.linalg.norm(Y_in, axis=1, keepdims=True)
        local_D = Xnormed @ Ynormed.T

    # ========================================
	# KNN computation
	# ========================================
        idx = np.argpartition(-local_D, n_neighbors - 1, axis=1)[: , :n_neighbors]
        vals = np.take_along_axis(local_D, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        local_indices = np.take_along_axis(idx, order, axis=1)
        local_scores = np.take_along_axis(vals, order, axis=1)

    # Gather them back
    indices = comm.gather(local_indices, root=0)
    scores = comm.gather(local_scores, root=0)
    if rank == 0:
        indices = np.vstack(indices)
        scores = np.vstack(scores)
        return(scores, indices)
    return(None, None)


# =============
# Begin Timing
# =============
comm.Barrier()
start = MPI.Wtime()

pympi_scrs, pympi_indxs = knn(X_in=X, Y_in=Y, n_neighbors=15)

comm.Barrier()
end = MPI.Wtime()
# =============
# End Timing
# =============
max_time = comm.reduce((end - start) * 1e6, op=MPI.MAX, root=0)

if rank == 0:
    py_scrs, py_indxs = skKnn()
    py_scrs = 1 - (py_scrs**2) / 2 # converts norm'd distance to cosine similiarity

    print(f"Score Matrices: {np.allclose(pympi_scrs, py_scrs)}")
    print(f"Index Matrices: {np.array_equal(pympi_indxs, py_indxs)}")
    print(f"time: {max_time} us")