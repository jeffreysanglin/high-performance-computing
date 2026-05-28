#!/bin/bash

#SBATCH --job-name=distance
#SBATCH --output=distance.out
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

# print some useful information
echo "My NetID:             ${SLURM_JOB_USER}"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Job name is:          ${SLURM_JOB_NAME}"
echo "Job submitted from:   ${SLURM_SUBMIT_DIR}"
echo "Date:                 $(date -u '+%Y-%m-%d %H:%M UTC')"
echo -e "\n"

PROFILE="${PROFILE:-0}"                 # 0 = off, 1 = on

# activate spack env
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu
# Activate your venv built from the Spack python
source ~/cs2050env/bin/activate

TILE_SIZE="${TILE_SIZE:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
BASE_ROWS="${BASE_ROWS:-1000}"
DIMS="${DIMS:-256}"

rm -rf "build_t${TILE_SIZE}"
cmake -B "build_t${TILE_SIZE}" . \
  -DCMAKE_BUILD_TYPE=Release \
  -DTILE_SIZE="${TILE_SIZE}"

cmake --build "build_t${TILE_SIZE}" || exit 1

EXE="./build_t${TILE_SIZE}/distance"

# nvcc -O3 distance.cu -o distance -lopenblas
# Run the tasks for each thread count
echo "STRONG SCALING====================="
echo "==================================="
echo ""
for block in 64 128 256 512 1024; do
  echo "=============================="
  echo "STRONG: block_size=${block}, rows=${BASE_ROWS}, dims=${DIMS}, TILE_SIZE=${TILE_SIZE}"
  echo "=============================="

  if [ "$PROFILE" = "1" ] ; then
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --stats=true \
      --force-overwrite=true \
      -o "knn_cuda_strong_b${block}" \
      "${EXE}" "${block}" "${BASE_ROWS}" "${DIMS}"
  else
    "${EXE}" "${block}" "${BASE_ROWS}" "${DIMS}"
  fi
done

# Keep the block size constant, increase rows, and cuda will increase number of blocks to match.
echo "Weak SCALING====================="
echo "==================================="
echo ""
for scale in 1 2 4 8 16; do
  rows=$(awk -v base="$BASE_ROWS" -v s="$scale" \
    'BEGIN { printf "%d", base * sqrt(s) }')

  echo "=============================="
  echo "WEAK: scale=${scale}, block_size=${BLOCK_SIZE}, rows=${rows}, dims=${DIMS}, TILE_SIZE=${TILE_SIZE}"  
  echo "=============================="

  if [ "$PROFILE" = "1" ] ; then
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --stats=true \
      --force-overwrite=true \
      -o "knn_cuda_weak_s${scale}" \
      "${EXE}" "${BLOCK_SIZE}" "${rows}" "${DIMS}"
  else
    "${EXE}" "${BLOCK_SIZE}" "${rows}" "${DIMS}"
  fi
done

echo "CUDA CONFIGURATION SCALING====================="
echo "==================================="

for tile in 8 16 32; do
  rm -rf "build_t${tile}"
  cmake -B "build_t${tile}" . \
    -DCMAKE_BUILD_TYPE=Release \
    -DTILE_SIZE="${tile}"

  cmake --build "build_t${tile}" || exit 1

  EXE="./build_t${tile}/distance"

  echo "=============================="
  echo "TILE_SIZE=${tile}, BLOCK_SIZE=${BLOCK_SIZE}, rows=${BASE_ROWS}, dims=${DIMS}"
  echo "=============================="

  "${EXE}" "${BLOCK_SIZE}" "${BASE_ROWS}" "${DIMS}"
done