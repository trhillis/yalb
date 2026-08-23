#include "hello.h"
#include "lbm.h"
#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>


int main(int argc, char *argv[]) {
    int rank = 0, size = 1;

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);


    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    {

        int global_rows = 129;
        int global_cols = 129;
        int num_steps = 10000;

        if (argc >= 4) {
            global_rows = std::stoi(argv[1]);
            global_cols = std::stoi(argv[2]);
            std::cout << "Using grid size: " << global_rows << "x" << global_cols << "\n";
            num_steps = std::stoi(argv[3]);
            std::cout << "Using number of steps: " << num_steps << "\n";
        } else if (rank == 0) {
            std::cout << "Using default grid size: " << global_rows << "x" << global_cols << "\n";
            std::cout << "To specify grid size, run as: ./executable <rows> <cols>\n";
        }

        int base_rows = global_rows / size; // Base number of rows for each process
        int local_rows = base_rows;
        int local_cols = global_cols; // All processes have the same number of columns

        int remainder = global_rows % size; // Calculate the remainder rows

        if (rank < remainder) {
            local_rows += 1; // Distribute the remainder rows among the first 'remainder' processes
        }

        int local_start = (global_rows / size) * rank + std::min(rank, remainder); // Calculate the starting row for each process

        LBM lbm_grid = create_lbm(local_rows + 2, local_cols); // +2 for ghost cells

        create_walls(lbm_grid, local_start, global_rows);

        initialize_eq_conditions(lbm_grid);

        compute_density(lbm_grid);

        double local_initial_mass =
            compute_local_fluid_mass(lbm_grid);

        double initial_mass = 0.0;

        MPI_Allreduce(
            &local_initial_mass,
            &initial_mass,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            std::cout
                << "Initial fluid mass = "
                << initial_mass
                << '\n';
        }

        auto get_global_mass = [&](LBM& grid) {
            compute_density(grid);

            const double local_mass =
                compute_local_fluid_mass(grid);

            double global_mass = 0.0;

            MPI_Allreduce(
                &local_mass,
                &global_mass,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD
            );

            return global_mass;
        };

        std::cout << "Rank " << rank << " owns " << local_rows << " rows\n";

        Kokkos::View<double***> previous_velocity(
            "previous_velocity",
            lbm_grid.rows,
            lbm_grid.cols,
            2
        );

        bool have_previous_velocity = false;

        const int residual_interval = 100;
        const double residual_tolerance = 1.0e-8;
        // const int num_steps = 10000;

        Kokkos::fence(); // Ensure all Kokkos operations are complete before starting the timer

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime(); // Start the timer for the solver

        int completed_steps = 0;

        constexpr bool diagnostics = true;

        for (int step = 0; step < num_steps; ++step) {
            
            // compute_density(lbm_grid);
            // compute_velocity(lbm_grid);
            
            // collision_step(lbm_grid);

            // Kokkos::fence(); // Ensure all Kokkos operations are complete before halo exchange

            exchange_halos(
                lbm_grid,
                rank,
                size
            );

            collision_and_stream(
                lbm_grid,
                0.1,
                local_start,
                global_rows
            );

            completed_steps++;

            if (diagnostics && (step + 1) % residual_interval == 0) {
                compute_density(lbm_grid);
                compute_velocity(lbm_grid);

                if (!have_previous_velocity) {
                    Kokkos::deep_copy(
                        previous_velocity,
                        lbm_grid.v
                    );

                    have_previous_velocity = true;

                    if (rank == 0) {
                        std::cout
                            << "Stored first velocity field at step "
                            << step + 1
                            << '\n';
                    }
                } else {
                    auto velocity_host =
                        Kokkos::create_mirror_view_and_copy(
                            Kokkos::HostSpace(),
                            lbm_grid.v
                        );

                    auto previous_velocity_host =
                        Kokkos::create_mirror_view_and_copy(
                            Kokkos::HostSpace(),
                            previous_velocity
                        );

                    auto wall_host =
                        Kokkos::create_mirror_view_and_copy(
                            Kokkos::HostSpace(),
                            lbm_grid.wall
                        );

                    double local_residual = 0.0;

                    Kokkos::parallel_reduce(
                        "VelocityResidual",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                            {1, 0},
                            {lbm_grid.rows - 1, lbm_grid.cols}
                        ),
                        KOKKOS_LAMBDA(
                            int row,
                            int col,
                            double& max_value
                        ) {
                            if (!lbm_grid.wall(row, col)) {

                                const double dv =
                                    lbm_grid.v(row, col, 0) -
                                    previous_velocity(row, col, 0);

                                const double du =
                                    lbm_grid.v(row, col, 1) -
                                    previous_velocity(row, col, 1);

                                const double value =
                                    sqrt(du * du + dv * dv);

                                if (value > max_value) {
                                    max_value = value;
                                }
                            }
                        },
                        Kokkos::Max<double>(local_residual)
                    );
                    
                    double global_residual = 0.0;

                    MPI_Allreduce(
                        &local_residual,
                        &global_residual,
                        1,
                        MPI_DOUBLE,
                        MPI_MAX,
                        MPI_COMM_WORLD
                    );

                    if (rank == 0) {
                        std::cout
                            << std::setprecision(17)
                            << "Step " << step + 1
                            << ", residual = "
                            << global_residual
                            << '\n';
                    }

                    Kokkos::deep_copy(
                        previous_velocity,
                        lbm_grid.v
                    );

                    if (global_residual < residual_tolerance) {
                        if (rank == 0) {
                            std::cout
                                << "Residual below tolerance at step "
                                << step + 1
                                << '\n';
                        }
                        // break;
                    }
                }
            }

            if (diagnostics && (step + 1) % 1000 == 0) {
                const double global_mass = get_global_mass(lbm_grid);

                const double absolute_drift = global_mass - initial_mass;

                const double relative_drift = absolute_drift / initial_mass;

                if (rank == 0){
                    std::cout
                        << std::setprecision(17)
                        << "Step " << step + 1
                        << ", mass = " << global_mass
                        << ", absolute drift = " << absolute_drift
                        << ", relative drift = " << relative_drift
                        << '\n';
                }

                compute_velocity(lbm_grid);

                write_output_rho(
                    lbm_grid,
                    "data/rho/output_rho_rank" +
                        std::to_string(rank) +
                        "_step" +
                        std::to_string(step + 1) +
                        ".txt",
                    local_start
                );

                write_output_velocity(
                    lbm_grid,
                    "data/v/output_velocity_rank" +
                        std::to_string(rank) +
                        "_step" +
                        std::to_string(step + 1) +
                        ".txt",
                    local_start
                );

            }
        }
        
        Kokkos::fence(); // Ensure all Kokkos operations are complete before measuring time

        double local_elapsed_time = MPI_Wtime() - start_time;

        double elapsed_time = 0.0;

        MPI_Reduce(
            &local_elapsed_time,
            &elapsed_time,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0) {

            double total_lattice_Updates =
                static_cast<double>(completed_steps) *
                static_cast<double>(global_rows) *
                static_cast<double>(global_cols);

            double MLUPS =
                total_lattice_Updates /
                (elapsed_time * 1.0e6);

            std::cout
                << std::setprecision(17)
                << "Total elapsed time = "
                << elapsed_time
                << " seconds\n"
                << "Performance = "
                << MLUPS
                << " MLUPS\n";

        }

        
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
