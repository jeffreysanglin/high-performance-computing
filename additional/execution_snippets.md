# Adding scikit on top of spack in login node. Used this to create the env
```bash
/shared/courseSharedFolders/161588outer/161588/spack/var/spack/environments/CS-2050/.spack-env/view/bin/python3.12 \
  -m venv ~/cs2050env_gpu
```

# Sourcing the new env
```bash
source ~/cs2050env_gpu/bin/activate
```

# Installing the new wheel and pykokkos
```bash
python -c "print('python ok')"
python -m pip --version
python -m pip install --upgrade pip setuptools wheel

python -m pip install scikit-learn mpi4py 

cd ./pykokkos
python install_base.py install --verbose -- \
  -DENABLE_VIEW_RANKS=3 \
  -DENABLE_OPENMP=ON \
  -DENABLE_THREADS=OFF \
  -DENABLE_CUDA=OFF
```

# added to slurm
```bash
source ~/cs2050env_gpu/bin/activate
python run.py
```