# Optimizing KNN + Cosine Similarity with HPC

This project explores how much performance can improve when a familiar data science workflow is rewritten for high-performance computing.

## MPI reduced runtime from 20.8 ms to 2.5 ms, while CUDA reached 1.36 ms on 1,000-row test case.

I implemented **k-nearest neighbors search using cosine similarity** in several ways:

* Serial C++
* OpenMP
* MPI
* CUDA
* Python with `mpi4py`

The goal was not just to make the code faster, but to understand *where* different parallel computing strategies help, where they add overhead, and how they compare to a standard data science baseline like scikit-learn.

## Why this project?

KNN and cosine similarity are common building blocks in data science, search, recommendation systems, clustering, and dimensionality reduction workflows like UMAP. They are also simple enough to make performance tradeoffs visible.

This made them a good test case for comparing CPU threading, distributed computing, GPU acceleration, and Python-based parallelism.

## Implementations

| Version    | Approach                                  |
| ---------- | ----------------------------------------- |
| Serial C++ | Baseline implementation using BLAS        |
| OpenMP     | Multithreaded CPU version                 |
| MPI        | Distributed version across multiple tasks |
| CUDA       | GPU implementation                        |
| mpi4py     | Python-based distributed version          |

Each version includes build/run scripts, benchmarking logic, and profiling support where relevant.

## Main results

The biggest improvement came from the **MPI** and **CUDA** versions.

### MPI strong scaling

The MPI version scaled well as more tasks were added:

| MPI Tasks | Runtime   |
| --------- | --------- |
| 1         | 20,770 µs |
| 2         | 13,618 µs |
| 4         | 6,954 µs  |
| 8         | 4,032 µs  |
| 16        | 2,529 µs  |

This was the strongest CPU-based result. It reduced runtime significantly by splitting rows across ranks and gathering only the final top-k results.

### CUDA

The CUDA implementation had the fastest absolute runtime for the tested problem sizes:

| Rows  | Runtime  |
| ----- | -------- |
| 1,000 | 1,363 µs |
| 1,414 | 2,289 µs |
| 2,000 | 3,862 µs |
| 2,828 | 7,003 µs |

The GPU version kept data on-device as much as possible, which helped limit transfer overhead.

## What I learned

A few practical lessons stood out:

* Simple multithreading helped, but scaling quickly hit diminishing returns.
* MPI performed better because each process worked on local rows and communicated only when needed.
* CUDA was fastest overall, but required more careful memory and kernel design.
* `mpi4py` was much easier to write than C++ MPI, but slower.
* Profiling tools like VTune and Nsight Systems were essential for finding bottlenecks.

The main bottleneck across implementations was still the cosine similarity calculation, especially the matrix multiplication and normalization work.

## Tech used

* C++
* Python
* NumPy
* scikit-learn
* pybind11
* OpenMP
* MPI
* mpi4py
* CUDA
* CMake
* SLURM
* Intel VTune
* NVIDIA Nsight Systems

## Takeaway

This project gave me hands-on experience moving a familiar data science algorithm through several levels of optimization: from serial code, to multithreaded CPU code, to distributed computing, to GPU acceleration.

Conclusion: performance depends heavily on the shape of the problem, communication overhead, memory movement, and how much work can stay local to each thread, rank, or device.
