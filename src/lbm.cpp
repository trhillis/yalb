#include "lbm.h"
#include <iostream>
#include <fstream>
#include <numbers>
#include <mpi.h>

LBM create_lbm(int rows, int cols) {
    LBM grid;
    grid.rows = rows;
    grid.cols = cols;

    grid.rho = Kokkos::View<double**>("rho", rows, cols);

    grid.f = Kokkos::View<double***, Kokkos::LayoutRight>(
        "f",
        rows,
        cols,
        9
    );

    grid.f_next = Kokkos::View<double***, Kokkos::LayoutRight>(
        "f_next",
        rows,
        cols,
        9
    );

    grid.v = Kokkos::View<double***>(
        "v",
        rows,
        cols,
        2
    );

    grid.wall = Kokkos::View<bool**>(
        "wall",
        rows,
        cols
    );

    const int halo_size = cols * 9; // Each row has cols * 9 distribution functions

    grid.send_lower =
        Kokkos::View<double*>(
            "send_lower",
            halo_size
        );

    grid.send_upper =
        Kokkos::View<double*>(
            "send_upper",
            halo_size
        );

    grid.recv_lower =
        Kokkos::View<double*>(
            "recv_lower",
            halo_size
        );

    grid.recv_upper =
        Kokkos::View<double*>(
            "recv_upper",
            halo_size
        );

    grid.send_lower_host =
    Kokkos::View<double*, Kokkos::CudaHostPinnedSpace>(
        "send_lower_host", halo_size);

    grid.send_upper_host =
        Kokkos::View<double*, Kokkos::CudaHostPinnedSpace>(
            "send_upper_host", halo_size);

    grid.recv_lower_host =
        Kokkos::View<double*, Kokkos::CudaHostPinnedSpace>(
            "recv_lower_host", halo_size);

    grid.recv_upper_host =
        Kokkos::View<double*, Kokkos::CudaHostPinnedSpace>(
            "recv_upper_host", halo_size);

    return grid;
}

double compute_local_fluid_mass(const LBM& lbm)
{
    double local_mass = 0.0;

    Kokkos::parallel_reduce(
        "FluidMass",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),
        KOKKOS_LAMBDA(int row, int col, double& mass) {
            if (!lbm.wall(row, col)) {
                mass += lbm.rho(row, col);
            }
        },
        local_mass
    );

    return local_mass;
}

void print_lbm_message() {
    std::cout << "LBM initialized\n";
}

void create_walls(LBM& lbm, int local_start, int global_rows) {
    // Set walls on the left, right, and bottom boundaries of the grid
    // Need to ensure that each rank sets its own walls correctly based on its local grid portion
    Kokkos::parallel_for(
        "CreateWalls",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {lbm.rows, lbm.cols}),
        KOKKOS_LAMBDA(int row, int col) {
            
            const int global_row = local_start + row - 1;

            const bool ghost_row =
                row == 0 || row == lbm.rows - 1;

            const bool bottom_wall = 
                global_row == 0;

            const bool top_wall =
                global_row == global_rows - 1;

            const bool left_wall = 
                col == 0;

            const bool right_wall =
                col == lbm.cols - 1;
            
            if (ghost_row) {
                /*
                    Ghost rows represent neighboring rows, but their
                    first and last columns are still side walls.
                */
               lbm.wall(row, col) = 
                    left_wall || right_wall;
            } else {
                lbm.wall(row, col) =
                    bottom_wall ||
                    top_wall ||
                    left_wall ||
                    right_wall;
            }
        }
    );

}

void compute_density(LBM& lbm) {
    // Example computation: compute density from distribution functions
    Kokkos::parallel_for(
        "ComputeDensity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            double local_rho = 0.0;
            for (int i = 0; i < 9; ++i) {
                local_rho += lbm.f(x, y, i);
            }
            lbm.rho(x, y) = local_rho;
        }
    );
}

void compute_velocity(LBM& lbm) {
    // Example computation: compute velocity from distribution functions
    Kokkos::parallel_for(
        "ComputeVelocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),
        KOKKOS_LAMBDA(int row, int col) {
            double velocity_vertical = 0.0;
            double velocity_horizontal = 0.0;
            int drow[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; // Example velocity directions
            int dcol[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
            for (int i = 0; i < 9; ++i) {
                velocity_vertical += lbm.f(row, col, i) * drow[i];
                velocity_horizontal += lbm.f(row, col, i) * dcol[i];
            }
            if (lbm.rho(row, col) > 0) { // Avoid division by zero
                lbm.v(row, col, 0) = velocity_vertical / lbm.rho(row, col);
                lbm.v(row, col, 1) = velocity_horizontal / lbm.rho(row, col);
            } else {
                lbm.v(row, col, 0) = 0.0;
                lbm.v(row, col, 1) = 0.0;
            }
        }
    );
}

void write_output_f(const LBM& lbm, const std::string& filename) {
    Kokkos::View<double***, Kokkos::LayoutRight>::HostMirror f_host = Kokkos::create_mirror_view(lbm.f);
    Kokkos::deep_copy(f_host, lbm.f);

    std::ofstream output_file(filename);

    for (int x = 0; x < lbm.rows; ++x) {
        for (int y = 0; y < lbm.cols; ++y) {
            for (int i = 0; i < 9; ++i) {
                if (f_host(x, y, i) != 0.0) {
                    output_file << x << " "
                                << y << " "
                                << i << " "
                                << f_host(x, y, i) << "\n";
                }
            }
            output_file << "\n";
        }
    }
    output_file.close();
}

void write_output_rho(const LBM& lbm, const std::string& filename, int local_start) {
    Kokkos::View<double**>::HostMirror rho_host = Kokkos::create_mirror_view(lbm.rho);
    Kokkos::deep_copy(rho_host, lbm.rho);

    std::ofstream output_file(filename);

    for (int x = 1; x < lbm.rows - 1; ++x) {
        int global_x = local_start + (x - 1); // Calculate the global x-coordinate based on the local start index
        for (int y = 0; y < lbm.cols; ++y) {
            output_file << global_x << " "
                        << y << " "
                        << rho_host(x, y) << "\n";
        }
        output_file << "\n";
    }
    output_file.close();
}

void write_output_velocity(const LBM& lbm, const std::string& filename, int local_start) {
    auto v_host = Kokkos::create_mirror_view(lbm.v);
    Kokkos::deep_copy(v_host, lbm.v);

    std::ofstream output_file(filename);

    for (int row = 1; row < lbm.rows - 1; ++row) {
        const int global_row = local_start + (row - 1); // Calculate the global x-coordinate based on the local start index
        
        for (int col = 0; col < lbm.cols; ++col) {
            
            const double vertical_velocity = v_host(row, col, 0);

            const double horizontal_velocity = v_host(row, col, 1);

            output_file << global_row << " "
                        << col << " "
                        << horizontal_velocity << " "
                        << vertical_velocity << "\n";
        }
        output_file << "\n";
    }
    output_file.close();
}

Kokkos::View<double***, Kokkos::LayoutRight> compute_equilibrium(LBM& lbm) {
    Kokkos::View<double***, Kokkos::LayoutRight> feq("feq", lbm.rows, lbm.cols, 9);

    double w[9] = {
        4.0/9.0,
        1.0/9.0,
        1.0/9.0,
        1.0/9.0,
        1.0/9.0,
        1.0/36.0,
        1.0/36.0,
        1.0/36.0,
        1.0/36.0
    };

    int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

    Kokkos::parallel_for(
        "ComputeEquilibrium",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            double rho = lbm.rho(x, y);
            double vx = lbm.v(x, y, 0);
            double vy = lbm.v(x, y, 1);

            for (int i = 0; i < 9; ++i) {
                double cu = cx[i] * vx + cy[i] * vy;
                double v2 = vx*vx + vy*vy;
                feq(x, y, i) = w[i] * rho *
                    (1 + 3*cu + 4.5*cu*cu - 1.5*v2);
            }
        }
    );

    return feq;
}

double compute_total_mass(const LBM& lbm) {
    auto rho_host = Kokkos::create_mirror_view(lbm.rho);
    Kokkos::deep_copy(rho_host, lbm.rho);
    
    double mass = 0.0;
    for (int x = 1; x < lbm.rows - 1; ++x) {
        for (int y = 0; y < lbm.cols; ++y) {
            mass += rho_host(x, y);
        }
    }

    return mass;
}

void initialize_eq_conditions(LBM& lbm){
    // rho = 1.0 everywhere
    // v = 0 everywhere
    // f = feq(rho, v) everywhere
    Kokkos::parallel_for(
        "InitializeFixedPoint",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {lbm.rows - 1, lbm.cols}),
        KOKKOS_LAMBDA(int x, int y) {
            lbm.rho(x, y) = 1.0; // uniform density
            lbm.v(x, y, 0) = 0.0; // vx
            lbm.v(x, y, 1) = 0.0; // vy
        }
    );
    auto feq = compute_equilibrium(lbm);


    Kokkos::parallel_for(
        "InitializeDistributionFunctions",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 0, 0}, {lbm.rows - 1, lbm.cols, 9}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            lbm.f(x, y, i) = feq(x, y, i);
        }
    );
    
}


void exchange_halos(LBM& lbm, int rank, int size) {
    
    if (size == 1) {
        // No need to exchange halos if there's only one rank

        static bool printed = false;

        if (!printed) {
            std::cout << "Skipping halo exchange for single rank\n";
            printed = true;
        }
        
        return;
    }

    exchange_halos_host(lbm, rank, size);

}

void exchange_halos_host(LBM& lbm, int rank, int size)
{
    const int lower_rank =
        (rank == 0) ? MPI_PROC_NULL : rank - 1;

    const int upper_rank =
        (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    const int n = lbm.cols * 9;

    Kokkos::parallel_for(
        "PackHalos",
        Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(int idx) {
            const int col = idx / 9;
            const int i   = idx % 9;

            lbm.send_lower(idx) = lbm.f(1, col, i);
            lbm.send_upper(idx) = lbm.f(lbm.rows - 2, col, i);
        }
    );

    if (lower_rank != MPI_PROC_NULL) {
        Kokkos::deep_copy(
            lbm.send_lower_host,
            lbm.send_lower
        );
    }

    if (upper_rank != MPI_PROC_NULL) {
        Kokkos::deep_copy(
            lbm.send_upper_host,
            lbm.send_upper
        );
    }

    MPI_Sendrecv(
        lbm.send_lower_host.data(),
        n,
        MPI_DOUBLE,
        lower_rank,
        100,

        lbm.recv_upper_host.data(),
        n,
        MPI_DOUBLE,
        upper_rank,
        100,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        lbm.send_upper_host.data(),
        n,
        MPI_DOUBLE,
        upper_rank,
        200,

        lbm.recv_lower_host.data(),
        n,
        MPI_DOUBLE,
        lower_rank,
        200,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    if (lower_rank != MPI_PROC_NULL) {
        Kokkos::deep_copy(
            lbm.recv_lower,
            lbm.recv_lower_host
        );
    }

    if (upper_rank != MPI_PROC_NULL) {
        Kokkos::deep_copy(
            lbm.recv_upper,
            lbm.recv_upper_host
        );
    }

    Kokkos::parallel_for(
        "UnpackHalos",
        Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(int idx) {
            const int col = idx / 9;
            const int i   = idx % 9;

            if (lower_rank != MPI_PROC_NULL) {
                lbm.f(0, col, i) =
                    lbm.recv_lower(idx);
            }

            if (upper_rank != MPI_PROC_NULL) {
                lbm.f(lbm.rows - 1, col, i) =
                    lbm.recv_upper(idx);
            }
        }
    );
}

void collision_and_stream(
    LBM& lbm,
    double u_lid,
    int local_start,
    int global_rows)
{
    constexpr double tau = 0.596;
    constexpr double omega = 1.0 / tau;

    Kokkos::parallel_for(
        "CollisionAndStream",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {1, 0},
            {lbm.rows - 1, lbm.cols}
        ),
        KOKKOS_LAMBDA(int row, int col) {

            if (lbm.wall(row, col)) {
                return;
            }

            constexpr double w[9] = {
                4.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0
            };

            constexpr int cx[9] =
                {0, 1, 0, -1, 0, 1, -1, -1, 1};

            constexpr int cy[9] =
                {0, 0, 1, 0, -1, 1, 1, -1, -1};

            constexpr int opposite[9] =
                {0, 3, 4, 1, 2, 7, 8, 5, 6};

            double f[9];

            // -----------------------------------------
            // PULL STREAM
            // -----------------------------------------

            for (int i = 0; i < 9; ++i) {

                const int src_row = row - cx[i];
                const int src_col = col - cy[i];

                const bool source_outside =
                    src_row < 0 ||
                    src_row >= lbm.rows ||
                    src_col < 0 ||
                    src_col >= lbm.cols;

                // Outside the local domain:
                // ordinary bounce-back
                if (source_outside) {
                    f[i] = lbm.f(row, col, opposite[i]);
                    continue;
                }

                // Source is a physical wall
                if (lbm.wall(src_row, src_col)) {

                    // Convert this rank's local row to the corresponding
                    // row in the complete global grid.
                    const int src_global_row =
                        local_start + src_row - 1;

                    // Is this wall the moving top lid?
                    //
                    // Exclude the left/right corner nodes.
                    const bool hits_moving_lid =
                        src_global_row == global_rows - 1 &&
                        col > 0 &&
                        col < lbm.cols - 1;

                    if (hits_moving_lid) {

                        const double rho_wall =
                            lbm.rho(row, col);

                        const double wall_correction =
                            6.0 *
                            w[i] *
                            rho_wall *
                            cy[i] *
                            u_lid;

                        f[i] =
                            lbm.f(row, col, opposite[i]) +
                            wall_correction;

                    } else {

                        // Stationary wall
                        f[i] =
                            lbm.f(row, col, opposite[i]);
                    }

                    continue;
                }

                // Normal fluid source
                f[i] =
                    lbm.f(src_row, src_col, i);
            }

            // -----------------------------------------
            // MACROSCOPIC QUANTITIES
            // -----------------------------------------

            const double rho =
                f[0] + f[1] + f[2] +
                f[3] + f[4] + f[5] +
                f[6] + f[7] + f[8];

            const double ux =
                (
                    f[1] - f[3] +
                    f[5] - f[6] -
                    f[7] + f[8]
                ) / rho;

            const double uy =
                (
                    f[2] - f[4] +
                    f[5] + f[6] -
                    f[7] - f[8]
                ) / rho;

            const double u2 =
                ux * ux + uy * uy;

            // -----------------------------------------
            // COLLISION + WRITE DIRECTLY TO f_next
            // -----------------------------------------

            for (int i = 0; i < 9; ++i) {

                const double cu =
                    cx[i] * ux +
                    cy[i] * uy;

                const double feq =
                    w[i] *
                    rho *
                    (
                        1.0 +
                        3.0 * cu +
                        4.5 * cu * cu -
                        1.5 * u2
                    );

                lbm.f_next(row, col, i) =
                    f[i] -
                    omega * (f[i] - feq);
            }

            lbm.rho(row, col) = rho;
            lbm.v(row, col, 0) = ux;
            lbm.v(row, col, 1) = uy;
        }
    );

    Kokkos::fence();

    std::swap(lbm.f, lbm.f_next);
}