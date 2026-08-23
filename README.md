# YALB

## Clone

```bash
git clone https://github.com/trhillis/yalb.git
cd yalb
```

## Build on CPU

```bash
cmake -S . -B build
cmake --build build --target main -j $(nproc)
```

## Build on GPU — RTX 4060

Requires CUDA and `nvcc`.

```bash
cmake -S . -B build-gpu \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ARCH_ADA89=ON

cmake --build build-gpu --target main -j $(nproc)
```

## Build on Cluster — A100

```bash
module purge
module load toolkit/nvidia-hpc-sdk/25.1
module load mpi/openmpi/5.0-nvidia-25.1-nompi

cmake -S . -B build-gpu \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ARCH_AMPERE80=ON

cmake --build build-gpu --target main -j $(nproc)
```

## Run

Arguments:

```text
main <rows> <columns> <number_of_steps>
```

### CPU

```bash
mpirun -np 4 ./build/executables/main 300 300 10000
```

### GPU

```bash
mpirun -np 1 ./build-gpu/executables/main 300 300 10000
```

For multiple GPUs, use one MPI rank per GPU. On the cluster, request the required GPUs through Slurm before running.