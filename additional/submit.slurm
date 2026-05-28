#!/usr/bin/env bash
#SBATCH --job-name=distance
#SBATCH --output=distance.out
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:10:00

# =========================
# Config (override at submit time)
# =========================
PROFILE="${PROFILE:-0}"                 # 0 = off, 1 = on
VTUNE_TASKS="${VTUNE_TASKS:-1}"    # which MPI task count to profile
VTUNE_RESULT_DIR="${VTUNE_RESULT_DIR:-vtune_hotspots}"
ONCE="${ONCE:-0}"

# Create thread counts array
TASK_COUNTS=(1 2 4 8 16 24 48)
BASE_ROWS=1000

# print some useful information
echo "My NetID:             ${SLURM_JOB_USER}"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Job name is:          ${SLURM_JOB_NAME}"
echo "Job submitted from:   ${SLURM_SUBMIT_DIR}"
echo "Date:                 $(date -u '+%Y-%m-%d %H:%M UTC')"
echo -e "\n"
echo "How many CPUs on this node?     ${SLURM_CPUS_ON_NODE}"
echo "How many tasks did I request?   ${SLURM_NTASKS}"
echo "PROFILE:              ${PROFILE}"
echo -e "\n"

# activate spack env
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

MPI_PREFIX="$(dirname "$(dirname "$(which mpicc)")")"
export LD_LIBRARY_PATH="${MPI_PREFIX}/lib:${MPI_PREFIX}/lib64:${LD_LIBRARY_PATH}"

# echo "MPI_PREFIX=${MPI_PREFIX}"
# echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
find "${MPI_PREFIX}" -name 'libmpi.so*' 2>/dev/null

# Activate your venv built from the Spack python
source ~/cs2050env_gpu/bin/activate

export PMIX_MCA_gds=hash
export OMP_PROC_BIND=close # pack threads close together on nearby cores
export OMP_PLACES=cores # prevent multiple threads from sharing same core.


echo "STRONG SCALING====================="
echo "==================================="
for n in "${TASK_COUNTS[@]}"; do
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export OPENBLAS_NUM_THREADS=1

    echo "=============================="
    echo "MPI tasks=$n"
    echo "ROW_COUNT=$BASE_ROWS"    
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    echo "Total cores used: $((n * SLURM_CPUS_PER_TASK))"
    echo "=============================="

    if [ "$PROFILE" = "1" ] && [ "$n" = "$VTUNE_TASKS" ]; then
        srun --mpi=pmix \
            --ntasks="$n" \
            --cpus-per-task="$SLURM_CPUS_PER_TASK" \
            vtune -collect hotspots \
            -result-dir "${VTUNE_RESULT_DIR}_n${n}" \
            -knob enable-stack-collection=true \
            -- python distance_mpi4py.py "$BASE_ROWS"
    else
        srun --mpi=pmix \
            --ntasks="$n" \
            --cpus-per-task="$SLURM_CPUS_PER_TASK" \
            -- python distance_mpi4py.py "$BASE_ROWS"
    fi

    if [ "$ONCE" = "1" ]; then 
        break
    fi
    echo
done

echo "WEAK SCALING======================="
echo "==================================="
for n in "${TASK_COUNTS[@]}"; do
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export OPENBLAS_NUM_THREADS=1

    rows=$(awk -v base="$BASE_ROWS" -v p="$n" \
        'BEGIN { printf "%d", base * sqrt(p) }')

    echo "=============================="
    echo "MPI tasks=$n"
    echo "ROW_COUNT=$rows"
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    echo "Total cores used: $((n * SLURM_CPUS_PER_TASK))"
    echo "=============================="

    if [ "$PROFILE" = "1" ] && [ "$n" = "$VTUNE_TASKS" ]; then
        srun --mpi=pmix \
            --ntasks="$n" \
            --cpus-per-task="$SLURM_CPUS_PER_TASK" \
            vtune -collect hotspots \
            -result-dir "${VTUNE_RESULT_DIR}_n${n}" \
            -knob enable-stack-collection=true \
            -- python distance_mpi4py.py "$rows"
    else
        srun --mpi=pmix \
            --ntasks="$n" \
            --cpus-per-task="$SLURM_CPUS_PER_TASK" \
            -- python distance_mpi4py.py "$rows"
    fi

    if [ "$ONCE" = "1" ]; then 
        break
    fi
    echo
done