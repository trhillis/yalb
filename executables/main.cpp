// Include the project's hello header file.
#include "hello.h"

// Include the main LBM declarations, including the LBM data structure
// and functions used to initialise, update, communicate, and output the simulation.
#include "lbm.h"

// Provides standard C++ input/output functionality such as std::cout.
#include <iostream>

// Provides C++ filesystem functionality.
#include <filesystem>

// Provides the MPI API used for distributed-memory parallelism.
#include <mpi.h>

// Provides the main Kokkos API used for portable CPU/GPU parallelism.
#include <Kokkos_Core.hpp>

// Provides standard algorithms such as std::min.
#include <algorithm>

// Provides mathematical functions such as sqrt.
#include <cmath>

// Provides stream formatting utilities such as std::setprecision.
#include <iomanip>


// Main entry point of the program.
// argc contains the number of command-line arguments.
// argv contains the command-line arguments themselves.
int main(int argc, char *argv[]) {

    // rank identifies this MPI process.
    // size stores the total number of MPI processes.
    // The initial values correspond to the single-process case before MPI
    // overwrites them below.
    int rank = 0, size = 1;

    // Initialise MPI.
    // argc and argv are passed so MPI can inspect/process relevant
    // command-line arguments if necessary.
    MPI_Init(&argc, &argv);

    // Initialise Kokkos.
    // This sets up the execution backend, which may be CPU or CUDA
    // depending on how Kokkos was compiled.
    Kokkos::initialize(argc, argv);


    // Determine the MPI rank of this process in MPI_COMM_WORLD.
    // Ranks are numbered from 0 to size - 1.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process

    // Determine how many MPI processes are participating in the simulation.
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    // Begin an additional scope so objects such as Kokkos Views are
    // destroyed before Kokkos::finalize() is called.
    {

        // Default number of rows in the complete/global simulation domain.
        int global_rows = 129;

        // Default number of columns in the complete/global simulation domain.
        int global_cols = 129;

        // Default maximum number of LBM timesteps to execute.
        int num_steps = 10000;

        // Check whether the user supplied three simulation arguments:
        // rows, columns, and number of timesteps.
        // argc is at least 4 because argv[0] is the executable name.
        if (argc >= 4) {

            // Convert the first command-line argument to an integer and
            // use it as the global number of rows.
            global_rows = std::stoi(argv[1]);

            // Convert the second command-line argument to an integer and
            // use it as the global number of columns.
            global_cols = std::stoi(argv[2]);

            // Report the selected global grid dimensions.
            std::cout << "Using grid size: " << global_rows << "x" << global_cols << "\n";

            // Convert the third command-line argument to an integer and
            // use it as the number of simulation timesteps.
            num_steps = std::stoi(argv[3]);

            // Report the selected number of timesteps.
            std::cout << "Using number of steps: " << num_steps << "\n";

        // If arguments were not supplied, only rank 0 prints the default
        // configuration so that every MPI rank does not print the same message.
        } else if (rank == 0) {

            // Report the default global grid dimensions.
            std::cout << "Using default grid size: " << global_rows << "x" << global_cols << "\n";

            // Tell the user how grid dimensions can be supplied.
            std::cout << "To specify grid size, run as: ./executable <rows> <cols>\n";
        }


        // A base row is the minimum number of global rows assigned to each MPI rank.
        // Calculate the minimum number of global rows assigned to each rank.
        // Integer division intentionally discards any remainder.
        int base_rows = global_rows / size; // Base number of rows for each process

        // Local rows are the actual number of rows assigned to this MPI rank
        // Start this rank with the base number of owned rows.
        int local_rows = base_rows;

        // Local_cols are the number of columns assigned to this MPI rank.
        // The domain is decomposed only in the row direction, so every
        // MPI rank owns the complete set of global columns.
        int local_cols = global_cols; // All processes have the same number of columns

        // Determine how many rows remain after equally dividing the rows.
        // These extra rows are distributed among the first few ranks.
        int remainder = global_rows % size; // Calculate the remainder rows

        // Ranks below 'remainder' receive one additional row.
        if (rank < remainder) {

            // Add the extra row to this rank's local domain.
            local_rows += 1; // Distribute the remainder rows among the first 'remainder' processes
        }

        // Calculate the first global row owned by this MPI rank.
        // std::min(rank, remainder) accounts for the extra rows assigned
        // to earlier ranks.
        int local_start = (global_rows / size) * rank + std::min(rank, remainder); // Calculate the starting row for each process

        // Create this rank's local LBM domain.
        // Two additional rows are allocated for lower and upper ghost/halo cells.
        // This is needed because the LBM streaming step requires data from neighboring rows.
        LBM lbm_grid = create_lbm(local_rows + 2, local_cols); // +2 for ghost cells

        // Mark physical wall cells in this rank's local portion of the domain.
        // local_start allows the function to determine where the local domain
        // lies within the complete global domain.
        // The walls will be created on the left, right, and bottom boundaries of the global domain.
        create_walls(lbm_grid, local_start, global_rows);

        // Initialise the particle distribution functions to their
        // equilibrium initial conditions.
        initialize_eq_conditions(lbm_grid);

        // Calculate the initial density field from the distribution functions.
        compute_density(lbm_grid);

        // Calculate the total fluid mass owned by this MPI rank.
        double local_initial_mass =
            compute_local_fluid_mass(lbm_grid);

        // Storage for the total mass across all MPI ranks.
        double initial_mass = 0.0;

        // Sum the local fluid masses from every MPI rank.
        // MPI_Allreduce returns the resulting global mass to every rank.
        MPI_Allreduce(

            // Address of this rank's local mass.
            &local_initial_mass,

            // Address where the global summed mass will be stored.
            &initial_mass,

            // Reduce one value.
            1,

            // The values being reduced are doubles.
            MPI_DOUBLE,

            // Sum the values from all ranks.
            MPI_SUM,

            // Perform the collective operation over every process.
            MPI_COMM_WORLD
        );

        // Only rank 0 prints the global initial mass.
        if (rank == 0) {

            // Print the initial total fluid mass.
            std::cout
                << "Initial fluid mass = "
                << initial_mass
                << '\n';
        }

        // Print the number of actual simulation rows owned by this rank.
        std::cout << "Rank " << rank << " owns " << local_rows << " rows\n";

        // Allocate a Kokkos View used to store the velocity field from
        // the previous residual measurement.
        // A Kokkos View is a multi-dimensional array that can be allocated in different memory spaces (e.g., CPU or GPU) and accessed in parallel.
        Kokkos::View<double***> previous_velocity(

            // Human-readable Kokkos allocation label.
            "previous_velocity",

            // Number of local rows, including ghost rows.
            lbm_grid.rows,

            // Number of local columns.
            lbm_grid.cols,

            // Two velocity components: x and y.
            2
        );

        // Indicates whether previous_velocity already contains a valid
        // velocity field against which the current field can be compared.
        bool have_previous_velocity = false;

        // Calculate the convergence residual every 100 timesteps.
        const int residual_interval = 100;

        // Residual below this value is considered to have met the
        // configured convergence tolerance.
        const double residual_tolerance = 1.0e-8;

        // Wait for any asynchronous Kokkos work to finish before timing
        // the solver so previous operations do not contaminate the timing.
        Kokkos::fence(); // Ensure all Kokkos operations are complete before starting the timer

        // Synchronise every MPI rank before starting the timer.
        // This prevents one rank from starting substantially earlier than another.
        MPI_Barrier(MPI_COMM_WORLD);

        // Record the solver start time using MPI's wall-clock timer.
        double start_time = MPI_Wtime(); // Start the timer for the solver

        // Count the number of timesteps that actually complete.
        int completed_steps = 0;

        // Compile-time switch controlling diagnostic calculations and output.
        constexpr bool diagnostics = true;

        // Main simulation timestep loop.
        // Execute at most num_steps iterations.
        for (int step = 0; step < num_steps; ++step) {

            // Exchange boundary/halo distribution data with neighbouring
            // MPI ranks so each local subdomain has the data required from
            // adjacent subdomains.
            // A halo is a layer of cells surrounding the local domain that stores
            // copies of neighboring data to facilitate computations that require neighbor information.
            exchange_halos(

                // This rank's LBM data.
                lbm_grid,

                // This process's MPI rank.
                rank,

                // Total number of MPI processes.
                size
            );

            // Perform the fused streaming and collision operation.
            collision_and_stream(

                // Local LBM domain.
                lbm_grid,

                // Velocity of the moving cavity lid.
                0.1,

                // Global index of this rank's first owned row.
                local_start,

                // Number of rows in the complete global domain.
                global_rows
            );

            // Record that another complete timestep has finished.
            completed_steps++;

            // Every residual_interval timesteps, perform convergence diagnostics.
            // This block is skipped entirely if diagnostics is false.
            if (diagnostics && (step + 1) % residual_interval == 0) {

                // Recalculate density for diagnostic purposes.
                compute_density(lbm_grid);

                // Recalculate velocity for diagnostic purposes.
                compute_velocity(lbm_grid);

                // The first residual checkpoint cannot calculate a difference
                // because there is no earlier stored velocity field.
                if (!have_previous_velocity) {

                    // Copy the current velocity field into previous_velocity.
                    Kokkos::deep_copy(

                        // Destination.
                        previous_velocity,

                        // Source.
                        lbm_grid.v
                    );

                    // Future residual checkpoints can now compare against
                    // the stored velocity field.
                    have_previous_velocity = true;

                    // Only rank 0 reports this event.
                    if (rank == 0) {

                        // Report the timestep at which the first reference
                        // velocity field was stored.
                        std::cout
                            << "Stored first velocity field at step "
                            << step + 1
                            << '\n';
                    }

                // A previous velocity field exists, so calculate the residual.
                } else {

                    // Create a host-space copy of the current velocity field.
                    // create_mirror_view_and_copy allocates a new Kokkos View in
                    // the specified memory space (here, Kokkos::HostSpace for CPU-accessible memory)
                    // and copies the data from the source View (lbm_grid.v) into it.
                    // basically mirrors GPU data to CPU memory for analysis.
                    auto velocity_host =
                        Kokkos::create_mirror_view_and_copy(

                            // Store the mirror in CPU-accessible host memory.
                            Kokkos::HostSpace(),

                            // Source velocity View.
                            lbm_grid.v
                        );

                    // Create a host-space copy of the previously stored velocity.
                    auto previous_velocity_host =
                        Kokkos::create_mirror_view_and_copy(

                            // Host memory destination.
                            Kokkos::HostSpace(),

                            // Source previous velocity View.
                            previous_velocity
                        );

                    // Create a host-space copy of the wall mask.
                    auto wall_host =
                        Kokkos::create_mirror_view_and_copy(

                            // Host memory destination.
                            Kokkos::HostSpace(),

                            // Source wall View.
                            lbm_grid.wall
                        );

                    // Stores the maximum velocity change found on this rank.
                    double local_residual = 0.0;

                    // Perform a parallel reduction over the local domain
                    // to determine the maximum velocity change.
                    // Runs many loop iterations in parallel, each comparing the
                    // current and previous velocity fields at a specific cell.
                    Kokkos::parallel_reduce(

                        // Label for profiling/debugging.
                        "VelocityResidual",

                        // Two-dimensional Kokkos iteration policy.
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(

                            // Start at row 1 to exclude the lower ghost row.
                            {1, 0},

                            // Stop before lbm_grid.rows - 1 to exclude the
                            // upper ghost row while covering all columns.
                            {lbm_grid.rows - 1, lbm_grid.cols}
                        ),

                        // GPU/CPU-portable lambda executed for each cell.
                        KOKKOS_LAMBDA(

                            // Current local row.
                            int row,

                            // Current column.
                            int col,

                            // Running maximum for the reduction.
                            double& max_value
                        ) {

                            // Only fluid cells contribute to the velocity residual.
                            if (!lbm_grid.wall(row, col)) {

                                // Difference between the current and previous
                                // first stored velocity component.
                                const double dv =
                                    lbm_grid.v(row, col, 0) -
                                    previous_velocity(row, col, 0);

                                // Difference between the current and previous
                                // second stored velocity component.
                                const double du =
                                    lbm_grid.v(row, col, 1) -
                                    previous_velocity(row, col, 1);

                                // Euclidean magnitude of the velocity change
                                // at this lattice cell.
                                // Use euclidean distance to combine the changes in both velocity components
                                // into a single scalar value representing the overall change in velocity at this cell.
                                const double value =
                                    sqrt(du * du + dv * dv);

                                // Update the reduction value when this cell
                                // has a larger velocity change.
                                if (value > max_value) {

                                    // Store the new maximum.
                                    max_value = value;
                                }
                            }
                        },

                        // Configure the reduction to return the maximum value.
                        Kokkos::Max<double>(local_residual)
                    );
                    
                    // Storage for the largest residual across all MPI ranks.
                    double global_residual = 0.0;

                    // Find the maximum local residual across the distributed domain.
                    // Use  the "worst" case residual from all ranks to determine convergence
                    MPI_Allreduce(

                        // This rank's maximum residual.
                        &local_residual,

                        // Destination for the global maximum residual.
                        &global_residual,

                        // Reduce one value.
                        1,

                        // Residual is represented as a double.
                        MPI_DOUBLE,

                        // Select the maximum value rather than summing.
                        MPI_MAX,

                        // Include every MPI rank.
                        MPI_COMM_WORLD
                    );

                    // Only rank 0 reports the global residual.
                    if (rank == 0) {

                        // Print the timestep and residual using high
                        // floating-point precision.
                        std::cout
                            << std::setprecision(17)
                            << "Step " << step + 1
                            << ", residual = "
                            << global_residual
                            << '\n';
                    }

                    // Replace the stored previous velocity field with the
                    // current velocity field for the next residual comparison.
                    // Use deep_copy to have separate memory spaces for previous_velocity and lbm_grid.v,
                    // ensuring that the data is copied correctly between them.
                    Kokkos::deep_copy(

                        // Destination.
                        previous_velocity,

                        // Current velocity field.
                        lbm_grid.v
                    );

                    // Test whether the global residual satisfies the
                    // configured convergence tolerance.
                    if (global_residual < residual_tolerance) {

                        // Only rank 0 reports convergence.
                        if (rank == 0) {

                            // Report when the residual first falls below tolerance.
                            std::cout
                                << "Residual below tolerance at step "
                                << step + 1
                                << '\n';
                        }

                        // Breaking here would stop the simulation immediately
                        // after reaching the convergence tolerance.
                        // It is currently disabled, so execution continues.
                        // break;
                    }
                }
            }

            // Every 1000 timesteps, write fields.
            // This will create output files filled with the density and velocity fields for each rank.
            if (diagnostics && (step + 1) % 1000 == 0) {

                // Calculate the velocity field before writing it to disk.
                compute_velocity(lbm_grid);

                // Write this rank's density field to a timestep-specific file.
                write_output_rho(

                    // Local LBM domain.
                    lbm_grid,

                    // Construct a filename containing both MPI rank and timestep.
                    "data/rho/output_rho_rank" +

                        // Add this MPI rank to the filename.
                        std::to_string(rank) +

                        // Separate the rank and timestep portions.
                        "_step" +

                        // Add the current one-based timestep.
                        std::to_string(step + 1) +

                        // Text-file extension.
                        ".txt",

                    // Supply the rank's global starting row so output can
                    // be associated with the correct global coordinates.
                    local_start
                );

                // Write this rank's velocity field to a timestep-specific file.
                write_output_velocity(

                    // Local LBM domain.
                    lbm_grid,

                    // Begin constructing the velocity output filename.
                    "data/v/output_velocity_rank" +

                        // Add MPI rank.
                        std::to_string(rank) +

                        // Separate rank from timestep.
                        "_step" +

                        // Add current timestep.
                        std::to_string(step + 1) +

                        // Text-file extension.
                        ".txt",

                    // Global starting row of this rank's subdomain.
                    local_start
                );

            }
        }
        
        // Ensure all asynchronous Kokkos operations have completed before
        // stopping the solver timer.
        Kokkos::fence(); // Ensure all Kokkos operations are complete before measuring time

        // Calculate this rank's wall-clock solver runtime.
        double local_elapsed_time = MPI_Wtime() - start_time;

        // Storage for the final solver runtime used for reporting.
        double elapsed_time = 0.0;

        // Determine the slowest rank's elapsed time.
        // Parallel execution cannot be considered complete until the slowest
        // participating rank has completed its work.
        MPI_Reduce(

            // This rank's elapsed time.
            &local_elapsed_time,

            // Rank 0 receives the final maximum elapsed time.
            &elapsed_time,

            // Reduce one value per rank.
            1,

            // Timing values are doubles.
            MPI_DOUBLE,

            // Select the maximum elapsed time across ranks.
            MPI_MAX,

            // Rank 0 is the root process receiving the result.
            0,

            // Include all MPI ranks.
            MPI_COMM_WORLD
        );

        // Only rank 0 calculates and prints the final performance statistics.
        if (rank == 0) {

            // Calculate the total number of lattice-site updates performed
            // over the entire global simulation.
            double total_lattice_Updates =

                // Number of completed timesteps.
                static_cast<double>(completed_steps) *

                // Number of rows in the global lattice.
                static_cast<double>(global_rows) *

                // Number of columns in the global lattice.
                static_cast<double>(global_cols);

            // Calculate performance in millions of lattice updates per second.
            double MLUPS =

                // Total number of lattice updates performed.
                total_lattice_Updates /

                // Divide by elapsed seconds and one million to convert
                // lattice updates/second into MLUPS.
                (elapsed_time * 1.0e6);

            // Print the measured solver runtime and resulting MLUPS.
            std::cout

                // Use high precision for the reported values.
                << std::setprecision(17)

                // Print elapsed-time label.
                << "Total elapsed time = "

                // Print maximum solver runtime across MPI ranks.
                << elapsed_time

                // Finish the elapsed-time line.
                << " seconds\n"

                // Print performance label.
                << "Performance = "

                // Print calculated MLUPS.
                << MLUPS

                // Finish the performance line.
                << " MLUPS\n";

        }

        
    }

    // Shut down Kokkos after all Kokkos-managed objects in the scope above
    // have been destroyed.
    Kokkos::finalize();

    // Shut down the MPI environment after all MPI communication is complete.
    MPI_Finalize();

    // Return zero to indicate successful program termination.
    return 0;
}