#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>

// 2.5D Parallel Matrix Multiplication in C++ with MPI
// ---------------------------------------------------
// A: NxN, B: NxN, C: NxN (square matrices for simplicity)
// 3D grid of processors: dims = [p_side, p_side, c]

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Get world rank & size
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Parse command-line args: N (matrix dim), c (replication factor)
    if (argc != 3) {
        if (world_rank == 0)
            fprintf(stderr, "Usage: %s <matrix_dim_N> <replication_c>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int N = atoi(argv[1]);
    int c = atoi(argv[2]);

    // Validate P divisible by c, and compute p_side
    int P = world_size;
    if (P % c != 0) {
        if (world_rank == 0)
            fprintf(stderr, "Error: P=%d must be divisible by c=%d\n", P, c);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int p_side = (int)std::round(std::sqrt((double)P / c));
    if (p_side * p_side * c != P) {
        if (world_rank == 0)
            fprintf(stderr, "Error: P/c=%d not a perfect square\n", P/c);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Create 3D Cartesian communicator
    int dims[3] = {p_side, p_side, c};
    int periods[3] = {1, 1, 1};   // wraparound
    MPI_Comm comm3d;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm3d);

    // Get my coords in comm3d
    int rank3d;
    MPI_Comm_rank(comm3d, &rank3d);
    int coords[3];
    MPI_Cart_coords(comm3d, rank3d, 3, coords);
    int i = coords[0], j = coords[1], k = coords[2];

    // Allocate local tiles
    int tile_size = N / p_side;
    double* A_tile = new double[tile_size * tile_size];
    double* B_tile = new double[tile_size * tile_size];
    double* C_tile = new double[tile_size * tile_size];
    for (int idx = 0; idx < tile_size * tile_size; idx++)
        C_tile[idx] = 0.0;

    // On front face (k==0), initialize A and B tiles
    if (k == 0) {
        srand48(time(NULL) * (i * p_side + j + 1));
        for (int idx = 0; idx < tile_size * tile_size; idx++) {
            A_tile[idx] = drand48();
            B_tile[idx] = drand48();
        }
    }

    // Sub-communicator along k-dimension
    int remain_dims[3] = {0, 0, 1};
    MPI_Comm comm_k;
    MPI_Cart_sub(comm3d, remain_dims, &comm_k);

    // Broadcast initial tiles along k
    int root_k = 0;
    MPI_Bcast(A_tile, tile_size * tile_size, MPI_DOUBLE, root_k, comm_k);
    MPI_Bcast(B_tile, tile_size * tile_size, MPI_DOUBLE, root_k, comm_k);

    // Synchronize and start timing
    MPI_Barrier(comm3d);
    double t_start = MPI_Wtime();

    // 2D shift-and-multiply loop
    for (int t = 0; t < p_side; t++) {
        int srcA, dstA;
        MPI_Cart_shift(comm3d, 1, -1, &srcA, &dstA);
        MPI_Sendrecv_replace(A_tile, tile_size * tile_size, MPI_DOUBLE,
                             dstA, 0, srcA, 0, comm3d, MPI_STATUS_IGNORE);

        int srcB, dstB;
        MPI_Cart_shift(comm3d, 0, -1, &srcB, &dstB);
        MPI_Sendrecv_replace(B_tile, tile_size * tile_size, MPI_DOUBLE,
                             dstB, 0, srcB, 0, comm3d, MPI_STATUS_IGNORE);

        // Local multiply accumulate
        for (int ii = 0; ii < tile_size; ii++) {
            for (int jj = 0; jj < tile_size; jj++) {
                double sum = 0.0;
                for (int kk2 = 0; kk2 < tile_size; kk2++) {
                    sum += A_tile[ii * tile_size + kk2] * B_tile[kk2 * tile_size + jj];
                }
                C_tile[ii * tile_size + jj] += sum;
            }
        }
    }

    // Reduce C_tile along k-dimension using MPI_IN_PLACE at root
    int rank_k;
    MPI_Comm_rank(comm_k, &rank_k);
    int count = tile_size * tile_size;
    if (rank_k == root_k) {
        MPI_Reduce(MPI_IN_PLACE, C_tile, count, MPI_DOUBLE, MPI_SUM, root_k, comm_k);
    } else {
        MPI_Reduce(C_tile, nullptr, count, MPI_DOUBLE, MPI_SUM, root_k, comm_k);
    }

    // Synchronize and stop timing
    MPI_Barrier(comm3d);
    double t_end = MPI_Wtime();

    // Print timing on (0,0,0)
    if (i == 0 && j == 0 && k == 0 && rank3d == 0) {
        printf("2.5D MatMul: N=%d, P=%d, c=%d, time=%.6f sec\n", N, P, c, t_end - t_start);
    }

    // Cleanup
    delete[] A_tile;
    delete[] B_tile;
    delete[] C_tile;
    MPI_Comm_free(&comm_k);
    MPI_Comm_free(&comm3d);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
