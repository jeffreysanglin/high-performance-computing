#!/usr/bin/env bash
#SBATCH --job-name=distance
#SBATCH --output=distance.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00

# =========================
# Config (override at submit time)
# =========================
PROFILE="${PROFILE:-0}"                 # 0 = off, 1 = on
VTUNE_RESULT_DIR="${VTUNE_RESULT_DIR:-vtune_hotspots}"

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

echo "SERIAL        ====================="
echo "==================================="
echo ""

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"

if [ "$PROFILE" = "1" ]; then
    echo "Running with VTune profiling..."
    # Run each thread count with 1 task, then run vtune hotspots profiler, pushing it out as a file.
    # hw sampling samples timing based on cpu events like cycles or instructions.
    srun --ntasks=1 --cpus-per-task=1 \
    vtune -collect hotspots \
        -result-dir "${VTUNE_RESULT_DIR}" \
        -knob enable-stack-collection=true \
        -- ./build/distance_profile_serial
else
    srun --ntasks=1 --cpus-per-task=1 python run.py
fi