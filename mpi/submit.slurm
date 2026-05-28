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
spack env activate -p CS-2050

# Activate your venv built from the Spack python
source ~/cs2050env/bin/activate

# print out py version
echo "PATH=$PATH"
echo "which python: $(which python)"
echo "which cmake:  $(which cmake)"
echo "which c++:    $(which c++)"
python -c "import sys; print(sys.executable); print(sys.version)"

# Refresh the build.
rm -rf build

#Compile
cmake -B build . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPython_EXECUTABLE="$(which python)" \
    -DPYTHON_EXECUTABLE="$(which python)"
cmake --build build

export PMIX_MCA_gds=hash
export OMP_PROC_BIND=close # pack threads close together on nearby cores
export OMP_PLACES=cores # prevent multiple threads from sharing same core.


# Run the tasks for each thread count
echo "STRONG SCALING====================="
echo "==================================="
echo ""
for n in "${TASK_COUNTS[@]}"; do
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export OPENBLAS_NUM_THREADS=1

    echo "=============================="
    echo "MPI tasks=$n"
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
            -- ./build/distance_profile
    else
        srun --mpi=pmix \
            --ntasks="$n" \
            --cpus-per-task="$SLURM_CPUS_PER_TASK" \
            python run.py
    fi

    if [ "$ONCE" = "1" ]; then 
        break
    fi
    echo
done


echo "WEAK SCALING====================="
echo "==================================="
echo ""
for n in "${TASK_COUNTS[@]}"; do
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export OPENBLAS_NUM_THREADS=1

    echo "=============================="
    echo "MPI tasks=$n"
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
            -- ./build/distance_profile
    else
        rows=$(awk -v n="$n" 'BEGIN { printf "%d", 1000 * sqrt(n) }')
        srun --mpi=pmix \
            --ntasks="$n" \
            --cpus-per-task="$SLURM_CPUS_PER_TASK" \
            python run.py "$rows"
    fi

    if [ "$ONCE" = "1" ]; then 
        break
    fi
    echo
done
