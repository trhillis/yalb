// Prevent this header file from being included more than once
// in the same translation unit during compilation.
#pragma once

// Include the core Kokkos functionality.
// This provides Kokkos::View and the memory/execution-space abstractions
// used throughout the LBM implementation.
#include <Kokkos_Core.hpp>


// Structure containing the data required for one local LBM domain.
// When MPI is used, each MPI rank creates its own LBM structure
// representing its portion of the complete global lattice.
struct LBM {

    // Two-dimensional density field.
    // rho(row, col) stores the macroscopic fluid density
    // at a particular lattice cell.
    Kokkos::View<double**> rho;

    // Main D2Q9 distribution-function field.
    // Each lattice cell contains nine distribution functions,
    // one for each D2Q9 lattice direction.
    //
    // LayoutRight means that the rightmost index is contiguous
    // in memory, so the nine distribution values belonging to
    // a cell are stored next to each other.
    Kokkos::View<double***, Kokkos::LayoutRight> f;

    // Secondary D2Q9 distribution-function field.
    // The fused collision-and-stream kernel writes the distributions
    // for the next timestep into this array.
    //
    // After the kernel finishes, f and f_next are swapped rather
    // than copying the entire distribution field.
    Kokkos::View<double***, Kokkos::LayoutRight> f_next;

    // Three-dimensional velocity field.
    // The first two indices identify the lattice cell.
    // The final index stores the two velocity components.
    //
    // v(row, col, 0) = first velocity component
    // v(row, col, 1) = second velocity component
    Kokkos::View<double***> v;


    // Two-dimensional boolean mask identifying wall cells.
    //
    // wall(row, col) == true  -> physical wall cell
    // wall(row, col) == false -> fluid cell
    Kokkos::View<bool**> wall;


    // Number of rows allocated for this local LBM domain.
    // For an MPI-decomposed domain, this includes the ghost/halo rows.
    int rows;

    // Number of columns in this local LBM domain.
    int cols;


    // Device-side buffer containing the packed distribution functions
    // from the first owned row that must be sent to the lower MPI rank.
    Kokkos::View<double*> send_lower;

    // Device-side buffer containing the packed distribution functions
    // from the final owned row that must be sent to the upper MPI rank.
    Kokkos::View<double*> send_upper;

    // Device-side buffer used to hold halo data received
    // from the lower neighbouring MPI rank.
    Kokkos::View<double*> recv_lower;

    // Device-side buffer used to hold halo data received
    // from the upper neighbouring MPI rank.
    Kokkos::View<double*> recv_upper;


    // Pinned host-memory staging buffer for the lower outgoing halo.
    //
    // The halo is first packed into send_lower on the GPU and then
    // copied here before being passed to MPI.
    //
    // CudaHostPinnedSpace allocates page-locked host memory, which
    // generally allows faster GPU-to-host transfers than ordinary
    // pageable host memory.
    Kokkos::View<double*, Kokkos::CudaHostPinnedSpace> send_lower_host;

    // Pinned host-memory staging buffer for the upper outgoing halo.
    //
    // Data is copied:
    //
    // send_upper (device)
    //      ->
    // send_upper_host (host)
    //      ->
    // MPI
    Kokkos::View<double*, Kokkos::CudaHostPinnedSpace> send_upper_host;

    // Pinned host-memory buffer into which MPI receives data
    // from the lower neighbouring rank.
    //
    // After MPI communication completes, this data is copied
    // into the device-side recv_lower buffer.
    Kokkos::View<double*, Kokkos::CudaHostPinnedSpace> recv_lower_host;

    // Pinned host-memory buffer into which MPI receives data
    // from the upper neighbouring rank.
    //
    // After MPI communication completes, this data is copied
    // into the device-side recv_upper buffer.
    Kokkos::View<double*, Kokkos::CudaHostPinnedSpace> recv_upper_host;
};

LBM create_lbm(int rows, int cols);
void print_lbm_message();
void stream_lbm(LBM& lbm, double u_lid, int local_start, int global_rows);
void compute_density(LBM& lbm);
void compute_velocity(LBM& lbm);
void write_output_f(const LBM& lbm, const std::string& filename);
void write_output_rho(const LBM& lbm, const std::string& filename, int local_start);
void write_output_velocity(const LBM& lbm, const std::string& filename, int local_start);
Kokkos::View<double***, Kokkos::LayoutRight> compute_equilibrium(LBM& lbm);
void collision_step(LBM& lbm);
void initialize_density_bump(LBM& lbm);
double compute_total_mass(const LBM& lbm);
void initialize_velocity_bump(LBM& lbm);
void initialize_fixed_point(LBM& lbm);
void initialize_shear_wave(LBM &lbm);
void create_walls(LBM &lbm, int local_start, int global_rows);
void initialize_eq_conditions(LBM& lbm);
void move_top_wall(LBM& lbm, double u_lid, int local_start, int global_rows);
void exchange_halos(LBM& lbm, int rank, int size);
void exchange_halos_host(LBM& lbm, int rank, int size);
void stream_lbm_pull(LBM& lbm, double u_lid, int local_start, int global_rows);
double compute_local_fluid_mass(const LBM& lbm);
void collision_and_stream(LBM& lbm, double u_lid, int local_start, int global_rows);
