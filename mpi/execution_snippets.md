# Adding scikit on top of spack in login node. Used this to create the env
```bash
/shared/courseSharedFolders/161588outer/161588/spack/var/spack/environments/CS-2050/.spack-env/view/bin/python3.12 -m venv ~/cs2050env
```

# Sourcing the new env
```bash
source ~/cs2050env/bin/activate
```

# Installing scikit like normal
```bash
pip install scikit-learn
```

# added to slurm
```bash
source ~/cs2050env/bin/activate
python run.py
```

# Run this with 1 task and 1 cpu per task.
```bash
sbatch --ntasks=1 --cpus-per-task=1 distance.slurm

sbatch --ntasks=1 --cpus-per-task=16 distance.slurm

sbatch --ntasks=8 --cpus-per-task=1 distance.slurm
```


## Notes from meeting with Chris:
How does performance scale across dimensions/embeddings?

maybe only find the top 10 neighbors to every point

analysis of the performance relative to the scale of the problem - strong scaling

umap library for python, does have a nearest neighbor calculation that will run under the hood. can give it the nearest neighbors matrix.

Most are not using euclidean distance; using cosin similarity, dot product of two vectors and the divided by product of the two vectors length

emphasize scaling plots, vtune, nsights