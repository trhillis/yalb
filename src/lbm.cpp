// Include the project's LBM header containing the LBM structure
// and declarations for the functions implemented in this file.
#include "lbm.h"

// Provides standard console input/output functionality.
#include <iostream>

// Provides file output functionality through std::ofstream.
#include <fstream>

// Provides standard mathematical constants such as pi.
#include <numbers>

// Provides MPI functionality for communication between ranks.
#include <mpi.h>


// Create and allocate all data required for an LBM subdomain.
// rows includes the two ghost/halo rows when MPI decomposition is used.
// cols is the number of columns in the local domain.
LBM create_lbm(int rows, int cols) {

    // Create the LBM structure that will hold all simulation data.
    LBM grid;

    // Store the number of local rows.
    grid.rows = rows;

    // Store the number of local columns.
    grid.cols = cols;

    // Allocate the two-dimensional density field.
    // One density value is stored for every lattice cell.
    grid.rho = Kokkos::View<double**>("rho", rows, cols);

    // Allocate the primary distribution-function field.
    // D2Q9 uses nine particle distribution values at every lattice cell.
    // LayoutRight makes the final index, the direction index here,
    // contiguous in memory.
    grid.f = Kokkos::View<double***, Kokkos::LayoutRight>(
        // Kokkos allocation label.
        "f",
        // Number of rows.
        rows,
        // Number of columns.
        cols,
        // Nine D2Q9 particle directions.
        9
    );

    // Allocate the second distribution-function field.
    // This is the destination during the fused streaming/collision operation
    // before f and f_next are swapped.
    grid.f_next = Kokkos::View<double***, Kokkos::LayoutRight>(
        // Kokkos allocation label.
        "f_next",
        // Number of rows.
        rows,
        // Number of columns.
        cols,
        // Nine D2Q9 directions.
        9
    );

    // Allocate the velocity field.
    // The final dimension stores the two velocity components.
    grid.v = Kokkos::View<double***>(
        // Kokkos allocation label.
        "v",
        // Number of rows.
        rows,
        // Number of columns.
        cols,
        // Two velocity components.
        2
    );

    // Allocate the wall mask.
    // Each cell stores true if it represents a physical wall.
    grid.wall = Kokkos::View<bool**>(
        // Kokkos allocation label.
        "wall",
        // Number of rows.
        rows,
        // Number of columns.
        cols
    );

    // Calculate how many distribution values are contained in one row.
    // Each column contains nine D2Q9 distributions.
    const int halo_size = cols * 9; // Each row has cols * 9 distribution functions

    // Allocate the device-side buffer used to pack the lower halo.
    grid.send_lower =
        Kokkos::View<double*>(
            // Kokkos allocation label.
            "send_lower",
            // One complete row of distribution values.
            halo_size
        );

    // Allocate the device-side buffer used to pack the upper halo.
    grid.send_upper =
        Kokkos::View<double*>(
            // Kokkos allocation label.
            "send_upper",
            // One complete row of distribution values.
            halo_size
        );

    // Allocate the device-side buffer into which the lower halo
    // received from another MPI rank will eventually be copied.
    grid.recv_lower =
        Kokkos::View<double*>(
            // Kokkos allocation label.
            "recv_lower",
            // One complete row of distribution values.
            halo_size
        );

    // Allocate the device-side buffer into which the upper halo
    // received from another MPI rank will eventually be copied.
    grid.recv_upper =
        Kokkos::View<double*>(
            // Kokkos allocation label.
            "recv_upper",
            // One complete row of distribution values.
            halo_size
        );

    // Allocate page-locked/pinned host memory for the lower outgoing halo.
    // Pinned memory generally provides more efficient CPU/GPU transfers
    // than ordinary pageable host memory.
    grid.send_lower_host =
    Kokkos::View<double*, Kokkos::CudaHostPinnedSpace>(
        // Kokkos allocation label and number of values.
        "send_lower_host", halo_size);

    // Allocate pinned host memory for the upper outgoing halo.
    grid.send_upper_host =
        Kokkos::View<double*, Kokkos::CudaHostPinnedSpace>(
            // Kokkos allocation label and number of values.
            "send_upper_host", halo_size);

    // Allocate pinned host memory for data received from the lower rank.
    grid.recv_lower_host =
        Kokkos::View<double*, Kokkos::CudaHostPinnedSpace>(
            // Kokkos allocation label and number of values.
            "recv_lower_host", halo_size);

    // Allocate pinned host memory for data received from the upper rank.
    grid.recv_upper_host =
        Kokkos::View<double*, Kokkos::CudaHostPinnedSpace>(
            // Kokkos allocation label and number of values.
            "recv_upper_host", halo_size);

    // Return the completely allocated LBM structure.
    return grid;
}


// Compute the total fluid mass belonging to this MPI rank.
// The const reference prevents the LBM object itself from being modified.
double compute_local_fluid_mass(const LBM& lbm)
{
    // Variable that will receive the result of the parallel reduction.
    double local_mass = 0.0;

    // Sum density over all non-wall cells owned by this rank.
    // parallel_reduce is basically a parallel for loop that also
    // computes a reduction (sum, max, etc.) over the iterations.
    Kokkos::parallel_reduce(

        // Profiling/debugging label for the Kokkos kernel.
        "FluidMass",

        // Iterate over the owned rows only.
        // Row 0 and rows-1 are ghost rows and therefore excluded.
        // MDRangePolicy allows for multi-dimensional iteration, here over 2D grid of rows and columns.
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),

        // Execute once for every owned lattice cell.
        // mass is Kokkos's reduction accumulator.
        KOKKOS_LAMBDA(int row, int col, double& mass) {

            // Only actual fluid cells contribute to fluid mass.
            if (!lbm.wall(row, col)) {

                // Add this cell's density to the rank-local total.
                mass += lbm.rho(row, col);
            }
        },

        // Store the completed reduction result here.
        local_mass
    );

    // Return the total fluid mass owned by this MPI rank.
    return local_mass;
}


// Print a simple message indicating that LBM functionality is available.
void print_lbm_message() {

    // Write the message to standard output.
    std::cout << "LBM initialized\n";
}


// Create the wall mask for this MPI rank's portion of the global domain.
void create_walls(LBM& lbm, int local_start, int global_rows) {

    // Set walls on the left, right, and bottom boundaries of the grid
    // Need to ensure that each rank sets its own walls correctly based on its local grid portion

    // Visit every local cell, including ghost rows, and determine
    // whether that location should be marked as a wall.
    Kokkos::parallel_for(

        // Profiling/debugging label.
        "CreateWalls",

        // Iterate over every row and column in the local allocation.
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),

        // Determine wall state for the current local cell.
        KOKKOS_LAMBDA(int row, int col) {
            
            // Convert this local row number to its corresponding global row.
            // Subtracting one accounts for the lower ghost row.
            const int global_row = local_start + row - 1;

            // Determine whether the current row is one of the two ghost rows.
            // Do this by checking if the local row is either the first or last row in the local allocation.
            const bool ghost_row =
                row == 0 || row == lbm.rows - 1;

            // The global row zero is the physical bottom wall.
            const bool bottom_wall = 
                global_row == 0;

            // The final global row is the physical top wall.
            const bool top_wall =
                global_row == global_rows - 1;

            // Column zero is the physical left wall.
            const bool left_wall = 
                col == 0;

            // The final column is the physical right wall.
            const bool right_wall =
                col == lbm.cols - 1;
            
            // Handle ghost rows separately from owned rows.
            if (ghost_row) {
                /*
                    Ghost rows represent neighboring rows, but their
                    first and last columns are still side walls.
                */

               // A ghost-row cell is considered a wall only when it also
               // lies on the left or right physical boundary.
               lbm.wall(row, col) = 
                    left_wall || right_wall;

            // Handle actual rows owned by this MPI rank.
            } else {

                // Mark the cell as a wall if it lies on any physical
                // boundary of the complete global domain.
                lbm.wall(row, col) =
                    bottom_wall ||
                    top_wall ||
                    left_wall ||
                    right_wall;
            }
        }
    );

}


// Compute density from the nine distribution functions at each owned cell.
void compute_density(LBM& lbm) {

    // Example computation: compute density from distribution functions

    // Run the density calculation in parallel over all owned cells.
    // The density calculation is the zeroth moment of the distribution functions:
    // density = sum(f_i) for i = 0 to 8, where f_i are the D2Q9 distribution functions.
    Kokkos::parallel_for(

        // Profiling/debugging label.
        "ComputeDensity",

        // Exclude the two ghost rows from the calculation.
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),

        // Calculate density at one lattice location.
        KOKKOS_LAMBDA(int x, int y) {

            // Start the local density accumulator at zero.
            double local_rho = 0.0;

            // Sum all nine D2Q9 distribution functions.
            for (int i = 0; i < 9; ++i) {

                // Density is the zeroth moment of the distribution functions.
                local_rho += lbm.f(x, y, i);
            }

            // Store the calculated density in the density field.
            lbm.rho(x, y) = local_rho;
        }
    );
}


// Compute the macroscopic velocity from the distribution functions.
void compute_velocity(LBM& lbm) {

    // Example computation: compute velocity from distribution functions

    // Run the velocity calculation over all owned lattice cells.
    // velocity is the first moment of the distribution functions divided by density:
    // velocity_x = sum(f_i * c_i_x) / rho
    // velocity_y = sum(f_i * c_i_y) / rho
    Kokkos::parallel_for(

        // Profiling/debugging label.
        "ComputeVelocity",

        // Exclude lower and upper ghost rows.
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),

        // Calculate velocity at one lattice cell.
        KOKKOS_LAMBDA(int row, int col) {

            // Accumulator for the vertical momentum component.
            double velocity_vertical = 0.0;

            // Accumulator for the horizontal momentum component.
            double velocity_horizontal = 0.0;

            // D2Q9 row-direction components for each distribution.
            int drow[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; // Example velocity directions

            // D2Q9 column-direction components for each distribution.
            int dcol[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

            // Accumulate momentum from all nine distributions.
            for (int i = 0; i < 9; ++i) {

                // Add this distribution's contribution to vertical momentum.
                velocity_vertical += lbm.f(row, col, i) * drow[i];

                // Add this distribution's contribution to horizontal momentum.
                velocity_horizontal += lbm.f(row, col, i) * dcol[i];
            }

            // Velocity is momentum divided by density.
            // Only divide if density is positive.
            if (lbm.rho(row, col) > 0) { // Avoid division by zero

                // Store the vertical velocity component.
                lbm.v(row, col, 0) = velocity_vertical / lbm.rho(row, col);

                // Store the horizontal velocity component.
                lbm.v(row, col, 1) = velocity_horizontal / lbm.rho(row, col);

            // If density is zero, explicitly set both components to zero.
            } else {

                // Set the first velocity component to zero.
                lbm.v(row, col, 0) = 0.0;

                // Set the second velocity component to zero.
                lbm.v(row, col, 1) = 0.0;
            }
        }
    );
}


// Write the complete distribution-function field to a text file.
void write_output_f(const LBM& lbm, const std::string& filename) {

    // Create a host-accessible mirror with the same dimensions/layout as f.
    Kokkos::View<double***, Kokkos::LayoutRight>::HostMirror f_host = Kokkos::create_mirror_view(lbm.f);

    // Copy the distribution data from its execution-space memory to host memory.
    Kokkos::deep_copy(f_host, lbm.f);

    // Open the requested output file.
    std::ofstream output_file(filename);

    // Iterate over every local row.
    for (int x = 0; x < lbm.rows; ++x) {

        // Iterate over every column.
        for (int y = 0; y < lbm.cols; ++y) {

            // Iterate over all nine D2Q9 distributions.
            for (int i = 0; i < 9; ++i) {

                // Only write non-zero distribution values.
                if (f_host(x, y, i) != 0.0) {

                    // Write row coordinate.
                    output_file << x << " "

                                // Write column coordinate.
                                << y << " "

                                // Write D2Q9 direction index.
                                << i << " "

                                // Write the distribution value.
                                << f_host(x, y, i) << "\n";
                }
            }

            // Separate lattice cells with a blank line.
            output_file << "\n";
        }
    }

    // Explicitly close the output file.
    output_file.close();
}


// Write this MPI rank's density field using global row coordinates.
void write_output_rho(const LBM& lbm, const std::string& filename, int local_start) {

    // Create a host-accessible mirror of the density field.
    Kokkos::View<double**>::HostMirror rho_host = Kokkos::create_mirror_view(lbm.rho);

    // Copy density from the execution space to host memory.
    Kokkos::deep_copy(rho_host, lbm.rho);

    // Open the requested output file.
    std::ofstream output_file(filename);

    // Iterate over owned rows only, excluding both ghost rows.
    for (int x = 1; x < lbm.rows - 1; ++x) {

        // Convert this rank-local row coordinate to a global row coordinate.
        int global_x = local_start + (x - 1); // Calculate the global x-coordinate based on the local start index

        // Iterate over all columns.
        for (int y = 0; y < lbm.cols; ++y) {

            // Write global row coordinate.
            output_file << global_x << " "

                        // Write column coordinate.
                        << y << " "

                        // Write density at this cell.
                        << rho_host(x, y) << "\n";
        }

        // Add a blank line between rows.
        output_file << "\n";
    }

    // Explicitly close the output file.
    output_file.close();
}


// Write this MPI rank's velocity field using global row coordinates.
void write_output_velocity(const LBM& lbm, const std::string& filename, int local_start) {

    // Create a host-accessible mirror of the velocity field.
    auto v_host = Kokkos::create_mirror_view(lbm.v);

    // Copy the velocity field into host-accessible memory.
    Kokkos::deep_copy(v_host, lbm.v);

    // Open the requested output file.
    std::ofstream output_file(filename);

    // Iterate over actual owned rows, excluding ghost rows.
    for (int row = 1; row < lbm.rows - 1; ++row) {

        // Convert the local row index into its global-domain row index.
        const int global_row = local_start + (row - 1); // Calculate the global x-coordinate based on the local start index
        
        // Iterate over every column.
        for (int col = 0; col < lbm.cols; ++col) {
            
            // Read the first stored velocity component.
            const double vertical_velocity = v_host(row, col, 0);

            // Read the second stored velocity component.
            const double horizontal_velocity = v_host(row, col, 1);

            // Write global row coordinate.
            output_file << global_row << " "

                        // Write column coordinate.
                        << col << " "

                        // Write horizontal velocity.
                        << horizontal_velocity << " "

                        // Write vertical velocity.
                        << vertical_velocity << "\n";
        }

        // Add a blank line between rows.
        output_file << "\n";
    }

    // Explicitly close the output file.
    output_file.close();
}


// Calculate the D2Q9 equilibrium distribution functions from the
// current density and velocity fields.
// This function returns a new Kokkos::View containing the equilibrium distributions.
// basically an array of size (rows, cols, 9) where each cell contains the equilibrium distribution for that cell.
// LayoutRight is used to ensure that the last dimension (the 9 directions) is contiguous in memory for better performance.
// i.e. feq(x, y, 0), feq(x, y, 1), ..., feq(x, y, 8) are stored contiguously in memory for each cell (x, y).
Kokkos::View<double***, Kokkos::LayoutRight> compute_equilibrium(LBM& lbm) {

    // Allocate a temporary field containing nine equilibrium
    // distributions for every lattice cell.
    Kokkos::View<double***, Kokkos::LayoutRight> feq("feq", lbm.rows, lbm.cols, 9);

    // D2Q9 lattice weights.
    double w[9] = {

        // Weight for the stationary distribution.
        4.0/9.0,

        // Weight for first cardinal direction.
        1.0/9.0,

        // Weight for second cardinal direction.
        1.0/9.0,

        // Weight for third cardinal direction.
        1.0/9.0,

        // Weight for fourth cardinal direction.
        1.0/9.0,

        // Weight for first diagonal direction.
        1.0/36.0,

        // Weight for second diagonal direction.
        1.0/36.0,

        // Weight for third diagonal direction.
        1.0/36.0,

        // Weight for fourth diagonal direction.
        1.0/36.0
    };

    // First components of the nine D2Q9 discrete velocity vectors.
    int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};

    // Second components of the nine D2Q9 discrete velocity vectors.
    int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

    // Compute equilibrium distributions for every owned lattice cell.
    Kokkos::parallel_for(

        // Profiling/debugging label.
        "ComputeEquilibrium",

        // Iterate over owned rows and every column.
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),

        // Calculate all nine equilibrium populations at one cell.
        KOKKOS_LAMBDA(int x, int y) {

            // Read density at this cell.
            double rho = lbm.rho(x, y);

            // Read the first velocity component.
            double vx = lbm.v(x, y, 0);

            // Read the second velocity component.
            double vy = lbm.v(x, y, 1);

            // Calculate equilibrium for each D2Q9 direction.
            for (int i = 0; i < 9; ++i) {

                // Calculate the dot product between the lattice velocity
                // direction and the local fluid velocity.
                double cu = cx[i] * vx + cy[i] * vy;

                // Calculate the squared magnitude of the fluid velocity.
                double v2 = vx*vx + vy*vy;

                // Evaluate the standard second-order D2Q9 equilibrium distribution.
                feq(x, y, i) = w[i] * rho *

                    // Equilibrium polynomial in velocity.
                    (1 + 3*cu + 4.5*cu*cu - 1.5*v2);
            }
        }
    );

    // Return the calculated equilibrium distribution field.
    return feq;
}


// Compute total local mass on the CPU from the current density field.
double compute_total_mass(const LBM& lbm) {

    // Create a host-accessible mirror of density.
    auto rho_host = Kokkos::create_mirror_view(lbm.rho);

    // Copy density from execution-space memory to host memory.
    Kokkos::deep_copy(rho_host, lbm.rho);
    
    // Initialise the mass accumulator.
    double mass = 0.0;

    // Iterate over owned rows, excluding ghost rows.
    for (int x = 1; x < lbm.rows - 1; ++x) {

        // Iterate over all columns.
        for (int y = 0; y < lbm.cols; ++y) {

            // Add this cell's density to the mass total.
            mass += rho_host(x, y);
        }
    }

    // Return the calculated mass.
    return mass;
}


// Initialise the LBM system at equilibrium with uniform density
// and zero velocity.
void initialize_eq_conditions(LBM& lbm){

    // rho = 1.0 everywhere
    // v = 0 everywhere
    // f = feq(rho, v) everywhere

    // Initialise the macroscopic density and velocity fields.
    Kokkos::parallel_for(

        // Profiling/debugging label.
        "InitializeFixedPoint",

        // Iterate over all owned cells.
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),

        // Initialise one lattice cell.
        KOKKOS_LAMBDA(int x, int y) {

            // Set uniform initial density to one.
            lbm.rho(x, y) = 1.0; // uniform density

            // Set the first initial velocity component to zero.
            lbm.v(x, y, 0) = 0.0; // vx

            // Set the second initial velocity component to zero.
            lbm.v(x, y, 1) = 0.0; // vy
        }
    );

    // Calculate equilibrium distributions corresponding to rho = 1
    // and zero initial velocity.
    auto feq = compute_equilibrium(lbm);


    // Copy the equilibrium distributions into the primary f field.
    Kokkos::parallel_for(

        // Profiling/debugging label.
        "InitializeDistributionFunctions",

        // Iterate over rows, columns, and all nine directions.
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0}, {lbm.rows - 1, lbm.cols, 9}),

        // Initialise one distribution value.
        KOKKOS_LAMBDA(int x, int y, int i) {

            // Set the distribution to its equilibrium value.
            lbm.f(x, y, i) = feq(x, y, i);
        }
    );
    
}


// Public halo-exchange function.
// Selects whether communication is necessary based on MPI process count.
void exchange_halos(LBM& lbm, int rank, int size) {
    
    // A single MPI rank has no neighbouring ranks with which to communicate.
    if (size == 1) {

        // No need to exchange halos if there's only one rank

        // Static means this value persists between function calls.
        // It prevents the informational message from being printed every timestep.
        static bool printed = false;

        // Print the message only on the first call.
        if (!printed) {

            // Inform the user that MPI halo communication is being skipped.
            std::cout << "Skipping halo exchange for single rank\n";

            // Prevent this message from being printed again.
            printed = true;
        }
        
        // Exit immediately because no halo communication is required.
        return;
    }

    // Perform halo exchange through host-pinned buffers.
    exchange_halos_host(lbm, rank, size);

}


// Exchange halo rows between neighbouring MPI ranks using
// host-pinned staging buffers.
void exchange_halos_host(LBM& lbm, int rank, int size)
{
    // Determine the rank below the current rank.
    // Rank zero has no lower neighbour, represented by MPI_PROC_NULL.
    const int lower_rank =
        (rank == 0) ? MPI_PROC_NULL : rank - 1;

    // Determine the rank above the current rank.
    // The final rank has no upper neighbour.
    const int upper_rank =
        (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Calculate the number of double values contained in one complete
    // D2Q9 lattice row.
    const int n = lbm.cols * 9;

    // Pack the first and last owned rows into contiguous device buffers.
    Kokkos::parallel_for(

        // Profiling/debugging label.
        "PackHalos",

        // Process every distribution value in one row.
        Kokkos::RangePolicy<>(0, n),

        // Convert the flat buffer index into column and direction indices.
        // We want to send the first owned row to the lower neighbour and 
        // the last owned row to the upper neighbour
        KOKKOS_LAMBDA(int idx) {

            // Determine which lattice column this flat index represents.
            // i.e. if idx = 0, col = 0; if idx = 9, col = 1; if idx = 18, col = 2; etc.
            const int col = idx / 9;

            // Determine which D2Q9 direction this flat index represents.
            // i.e. if idx = 0, i = 0; if idx = 1, i = 1; if idx = 8, i = 8; if idx = 9, i = 0; etc.
            const int i   = idx % 9;

            // Pack the first owned row into the lower outgoing buffer.
            // i.e. this looks like: send_lower[0] = f(1, 0, 0),
            // send_lower[1] = f(1, 0, 1), ..., send_lower[8] = f(1, 0, 8), send_lower[9] = f(1, 1, 0), ...,
            // send_lower[n-1] = f(1, cols-1, 8)
            lbm.send_lower(idx) = lbm.f(1, col, i);

            // Pack the final owned row into the upper outgoing buffer.
            // i.e. this looks like: send_upper[0] = f(rows-2, 0, 0),
            // send_upper[1] = f(rows-2, 0, 1), ..., send_upper[8] =
            // f(rows-2, 0, 8), send_upper[9] = f(rows-2, 1, 0), ..., send_upper[n-1] = f(rows-2, cols-1, 8)
            lbm.send_upper(idx) = lbm.f(lbm.rows - 2, col, i);
        }
    );

    // Only copy a lower outgoing halo if a lower MPI neighbour exists.
    if (lower_rank != MPI_PROC_NULL) {

        // Transfer the packed lower halo from device memory into
        // pinned host memory so non-CUDA-aware MPI can access it.
        Kokkos::deep_copy(

            // Host destination.
            lbm.send_lower_host,

            // Device source.
            lbm.send_lower
        );
    }

    // Only copy an upper outgoing halo if an upper MPI neighbour exists.
    if (upper_rank != MPI_PROC_NULL) {

        // Transfer the upper halo from device memory to pinned host memory.
        Kokkos::deep_copy(

            // Host destination.
            lbm.send_upper_host,

            // Device source.
            lbm.send_upper
        );
    }

    // Simultaneously send the lower owned row downward and receive
    // an upper ghost row from the upper neighbouring rank.
    MPI_Sendrecv(

        // Address of the lower outgoing host buffer.
        lbm.send_lower_host.data(),

        // Number of double values to send.
        n,

        // MPI datatype of the outgoing values.
        MPI_DOUBLE,

        // Destination rank below this rank.
        lower_rank,

        // Message tag for this communication direction.
        100,

        // Address where data received from the upper rank will be stored.
        lbm.recv_upper_host.data(),

        // Number of double values expected.
        n,

        // MPI datatype of incoming values.
        MPI_DOUBLE,

        // Receive from the upper neighbouring rank.
        upper_rank,

        // Matching communication tag.
        100,

        // Communicate over the global MPI communicator.
        MPI_COMM_WORLD,

        // Ignore detailed information about the completed receive.
        MPI_STATUS_IGNORE
    );

    // Simultaneously send the upper owned row upward and receive
    // a lower ghost row from the lower neighbouring rank.
    MPI_Sendrecv(

        // Address of the upper outgoing host buffer.
        lbm.send_upper_host.data(),

        // Number of values to send.
        n,

        // MPI datatype.
        MPI_DOUBLE,

        // Destination rank above this rank.
        upper_rank,

        // Tag distinguishing this exchange from the previous one.
        200,

        // Buffer receiving data from the lower neighbour.
        lbm.recv_lower_host.data(),

        // Number of expected values.
        n,

        // MPI datatype.
        MPI_DOUBLE,

        // Receive from the lower neighbouring rank.
        lower_rank,

        // Matching message tag.
        200,

        // Use the global MPI communicator.
        MPI_COMM_WORLD,

        // Detailed receive status is not required.
        MPI_STATUS_IGNORE
    );

    // If a lower neighbour exists, move its received data from
    // pinned host memory back into device-accessible memory.
    if (lower_rank != MPI_PROC_NULL) {

        // Copy lower received halo to device.
        Kokkos::deep_copy(

            // Device destination.
            lbm.recv_lower,

            // Host source.
            lbm.recv_lower_host
        );
    }

    // If an upper neighbour exists, move its received data back to the device.
    if (upper_rank != MPI_PROC_NULL) {

        // Copy upper received halo to device.
        Kokkos::deep_copy(

            // Device destination.
            lbm.recv_upper,

            // Host source.
            lbm.recv_upper_host
        );
    }

    // Unpack received contiguous halo buffers into the ghost rows of f.
    Kokkos::parallel_for(

        // Profiling/debugging label.
        "UnpackHalos",

        // Process every distribution value in the halo.
        Kokkos::RangePolicy<>(0, n),

        // Unpack one flat halo-buffer element.
        KOKKOS_LAMBDA(int idx) {

            // Recover column index.
            const int col = idx / 9;

            // Recover D2Q9 direction index.
            const int i   = idx % 9;

            // Only populate the lower ghost row if a lower neighbour exists.
            if (lower_rank != MPI_PROC_NULL) {

                // Write the received lower-neighbour data into ghost row zero.
                lbm.f(0, col, i) =
                    lbm.recv_lower(idx);
            }

            // Only populate the upper ghost row if an upper neighbour exists.
            if (upper_rank != MPI_PROC_NULL) {

                // Write the received upper-neighbour data into the final ghost row.
                lbm.f(lbm.rows - 1, col, i) =
                    lbm.recv_upper(idx);
            }
        }
    );
}


// Perform pull streaming and BGK collision together in one Kokkos kernel.
// u_lid is the velocity of the moving top wall.
// local_start identifies where this MPI rank begins in the global grid.
// global_rows is required to identify the physical top boundary.
void collision_and_stream(
    // LBM state being updated.
    LBM& lbm,

    // Prescribed moving-lid velocity.
    double u_lid,

    // First global row owned by this MPI rank.
    int local_start,

    // Total number of rows in the complete global domain.
    int global_rows)
{
    // BGK relaxation time.
    constexpr double tau = 0.5384;

    // Relaxation frequency, equal to the inverse of tau.
    constexpr double omega = 1.0 / tau;

    // Run one fused streaming/collision operation over the local domain.
    Kokkos::parallel_for(

        // Profiling/debugging label.
        "CollisionAndStream",

        // Two-dimensional iteration policy.
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(

            // Start at first owned row and first column.
            {1, 0},

            // Stop before the upper ghost row and process all columns.
            {lbm.rows - 1, lbm.cols}
        ),

        // Process one lattice cell.
        KOKKOS_LAMBDA(int row, int col) {

            // Physical wall cells are not collided as fluid cells.
            if (lbm.wall(row, col)) {

                // Stop processing this lattice location.
                return;
            }

            // Standard D2Q9 lattice weights.
            constexpr double w[9] = {

                // Rest population.
                4.0 / 9.0,

                // Cardinal population.
                1.0 / 9.0,

                // Cardinal population.
                1.0 / 9.0,

                // Cardinal population.
                1.0 / 9.0,

                // Cardinal population.
                1.0 / 9.0,

                // Diagonal population.
                1.0 / 36.0,

                // Diagonal population.
                1.0 / 36.0,

                // Diagonal population.
                1.0 / 36.0,

                // Diagonal population.
                1.0 / 36.0
            };

            // First coordinate components of the nine D2Q9 directions.
            constexpr int cx[9] =
                {0, 1, 0, -1, 0, 1, -1, -1, 1};

            // Second coordinate components of the nine D2Q9 directions.
            constexpr int cy[9] =
                {0, 0, 1, 0, -1, 1, 1, -1, -1};

            // Maps each D2Q9 direction to the direction pointing
            // exactly opposite to it, used for bounce-back boundaries.
            constexpr int opposite[9] =
                {0, 3, 4, 1, 2, 7, 8, 5, 6};

            // Temporary register/local array containing the nine populations
            // streamed into the current lattice cell.
            double f[9];

            // -----------------------------------------
            // PULL STREAM
            // -----------------------------------------

            // Pull each of the nine populations from the source cell
            // that points toward the current destination cell.
            for (int i = 0; i < 9; ++i) {

                // Calculate the row from which distribution i should be pulled.
                const int src_row = row - cx[i];

                // Calculate the column from which distribution i should be pulled.
                const int src_col = col - cy[i];

                // Determine whether the source location lies outside
                // this rank's allocated local domain.
                const bool source_outside =
                    src_row < 0 ||
                    src_row >= lbm.rows ||
                    src_col < 0 ||
                    src_col >= lbm.cols;

                // Outside the local domain:
                // ordinary bounce-back

                // Handle sources that lie outside the allocated local grid.
                if (source_outside) {

                    // Reflect the population by taking the opposite-direction
                    // population from the current cell.
                    f[i] = lbm.f(row, col, opposite[i]);

                    // This population has been handled; proceed to the next one.
                    continue;
                }

                // Source is a physical wall

                // Handle streaming from a cell marked as a physical wall.
                if (lbm.wall(src_row, src_col)) {

                    // Convert this rank's local row to the corresponding
                    // row in the complete global grid.

                    // This determines whether the wall belongs to the global
                    // moving top boundary or another stationary wall.
                    const int src_global_row =
                        local_start + src_row - 1;

                    // Is this wall the moving top lid?
                    //
                    // Exclude the left/right corner nodes.

                    // Identify the moving top-wall portion of the cavity.
                    const bool hits_moving_lid =
                        src_global_row == global_rows - 1 &&
                        col > 0 &&
                        col < lbm.cols - 1;

                    // Apply moving-wall bounce-back if the source is the lid.
                    if (hits_moving_lid) {

                        // Use the density of the adjacent fluid cell
                        // for the moving-wall correction.
                        const double rho_wall =
                            lbm.rho(row, col);

                        // Calculate the velocity correction associated with
                        // bounce-back from the moving lid.
                        const double wall_correction =

                            // D2Q9 moving-wall coefficient.
                            6.0 *

                            // Direction-specific lattice weight.
                            w[i] *

                            // Local fluid density.
                            rho_wall *

                            // Direction component associated with lid motion.
                            cy[i] *

                            // Prescribed lid velocity.
                            u_lid;

                        // Reflect the opposite population and add the
                        // moving-wall velocity correction.
                        f[i] =
                            lbm.f(row, col, opposite[i]) +
                            wall_correction;

                    // All other physical walls are stationary.
                    } else {

                        // Stationary wall

                        // Perform ordinary bounce-back by reflecting the
                        // opposite-direction population.
                        f[i] =
                            lbm.f(row, col, opposite[i]);
                    }

                    // Wall handling for this population is complete.
                    continue;
                }

                // Normal fluid source

                // Pull distribution i from its neighbouring fluid source cell.
                f[i] =
                    lbm.f(src_row, src_col, i);
            }

            // -----------------------------------------
            // MACROSCOPIC QUANTITIES
            // -----------------------------------------

            // Calculate density as the sum of all nine streamed populations.
            const double rho =
                f[0] + f[1] + f[2] +
                f[3] + f[4] + f[5] +
                f[6] + f[7] + f[8];

            // Calculate the first macroscopic velocity component
            // from the appropriate signed D2Q9 populations.
            // Positive contributions come from populations moving in the positive x-direction,
            // while negative contributions come from populations moving in the negative x-direction.
            const double ux =
                (
                    f[1] - f[3] +
                    f[5] - f[6] -
                    f[7] + f[8]
                ) / rho;

            // Calculate the second macroscopic velocity component.
            // Positive contributions come from populations moving in the positive y-direction,
            // while negative contributions come from populations moving in the negative y-direction.
            const double uy =
                (
                    f[2] - f[4] +
                    f[5] + f[6] -
                    f[7] - f[8]
                ) / rho;

            // Calculate squared velocity magnitude once so it can
            // be reused for all nine equilibrium distributions.
            const double u2 =
                ux * ux + uy * uy;

            // -----------------------------------------
            // COLLISION + WRITE DIRECTLY TO f_next
            // -----------------------------------------

            // Apply BGK collision independently to each D2Q9 population.
            for (int i = 0; i < 9; ++i) {

                // Dot product between the lattice direction and fluid velocity.
                // Measures how strongly the fluid is moving in the direction of the lattice velocity.
                const double cu =
                    cx[i] * ux +
                    cy[i] * uy;

                // Calculate the equilibrium distribution for direction i.
                const double feq =

                    // Direction-specific D2Q9 weight.
                    w[i] *

                    // Local density.
                    rho *

                    // Second-order equilibrium polynomial.
                    (
                        1.0 +
                        3.0 * cu +
                        4.5 * cu * cu -
                        1.5 * u2
                    );

                // Perform BGK relaxation toward equilibrium and write
                // directly into the next distribution field.
                lbm.f_next(row, col, i) =

                    // Current post-streaming distribution.
                    f[i] -

                    // Relax the difference between current and equilibrium
                    // distributions according to omega.
                    omega * (f[i] - feq);
            }

            // Store the newly calculated density for this fluid cell.
            lbm.rho(row, col) = rho;

            // Store the first newly calculated velocity component.
            lbm.v(row, col, 0) = ux;

            // Store the second newly calculated velocity component.
            lbm.v(row, col, 1) = uy;
        }
    );

    // Wait until the fused Kokkos kernel has completely finished before
    // swapping the old and new distribution fields.
    Kokkos::fence();

    // Swap the Kokkos View handles so f_next becomes the current distribution
    // field for the next timestep without copying the complete arrays.
    std::swap(lbm.f, lbm.f_next);
}