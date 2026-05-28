#!/usr/bin/env bash
#SBATCH --job-name=distance
#SBATCH --output=distance.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=00:15:00

# =========================
# Config (override at submit time)
# =========================
PROFILE="${PROFILE:-0}"                 # 0 = off, 1 = on
VTUNE_THREADS="${VTUNE_THREADS:-1}"    # which thread count to profile
VTUNE_RESULT_DIR="${VTUNE_RESULT_DIR:-vtune_hotspots}"
ONCE="${ONCE:-0}"

# Create thread counts array
THREAD_COUNTS=(1 2 4 8 16)

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

#Compile
rm -rf build

cmake -B build . \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g -fno-omit-frame-pointer" \
  -DPython_EXECUTABLE="$HOME/cs2050env/bin/python"

cmake --build build


# Run the tasks for each thread count
echo "STRONG SCALING====================="
echo "==================================="
echo ""
for t in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS="$t"
    export OPENBLAS_NUM_THREADS="$t"
    # Print section with thread counts for each.
    echo "=============================="
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    echo "=============================="
    
    # If the slurm was submitted with PROFILE=1 and it's single thread, then it will profile first thread.
    if [ "$PROFILE" = "1" ] && [ "$t" = "$VTUNE_THREADS" ]; then
        echo "Running with VTune profiling..."
        # Run each thread count with 1 task, then run vtune hotspots profiler, pushing it out as a file.
        # hw sampling samples timing based on cpu events like cycles or instructions.
        vtune -collect hotspots \
            -result-dir "${VTUNE_RESULT_DIR}_t${t}" \
            -knob enable-stack-collection=true \
            -- ./build/distance_profile
    else
        srun --ntasks=1 --cpus-per-task="$t" python run.py
    fi

    if [ "$ONCE" = "1" ]; then 
        break
    fi

    echo
done

echo "WEAK SCALING====================="
echo "==================================="
echo ""
# Run the tasks for each thread count
for t in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS="$t"
    export OPENBLAS_NUM_THREADS="$t"
    # Print section with thread counts for each.
    rows=$(awk -v t="$t" 'BEGIN { printf "%d", 1000 * sqrt(t) }')
    echo "=============================="
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    echo "ROW_COUNT=$((rows))"
    echo "=============================="
    
    # If the slurm was submitted with PROFILE=1 and it's single thread, then it will profile first thread.
    if [ "$PROFILE" = "1" ] && [ "$t" = "$VTUNE_THREADS" ]; then
        echo "Running with VTune profiling..."
        # Run each thread count with 1 task, then run vtune hotspots profiler, pushing it out as a file.
        # hw sampling samples timing based on cpu events like cycles or instructions.
        vtune -collect hotspots \
            -result-dir "${VTUNE_RESULT_DIR}_t${t}" \
            -knob enable-stack-collection=true \
            -- ./build/distance_profile
    else
        srun --ntasks=1 --cpus-per-task="$t" python run.py "$rows"
    fi

    if [ "$ONCE" = "1" ]; then 
        break
    fi

    echo
done