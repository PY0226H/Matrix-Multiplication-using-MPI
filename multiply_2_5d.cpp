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

    // 1. Parse command-line args: N (matrix dim), c (replication factor)
    if (argc != 3) {
        if (world_rank == 0)
            fprintf(stderr, "Usage: %s <matrix_dim_N> <replication_c>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int N = atoi(argv[1]);           // global matrix size
    int c = atoi(argv[2]);           // number of layers in z-dimension

    // 2. Validate that world_size P is divisible by c
    int P = world_size;
    if (P % c != 0) {
        if (world_rank == 0)
            fprintf(stderr, "Error: Number of ranks (P=%d) must be divisible by c (%d)\n", P, c);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // 3. Compute p_side = sqrt(P/c)
    int p_side = (int)std::round(std::sqrt(P / c));
    if (p_side * p_side * c != P) {
        if (world_rank == 0)
            fprintf(stderr, "Error: P/c=%d must be a perfect square (p_side^2)\n", P/c);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // 4. Create 3D Cartesian communicator
    int dims[3] = {p_side, p_side, c};
    int periods[3] = {1, 1, 1};   // wraparound in all dims for shifts
    MPI_Comm comm3d;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, /*reorder=*/0, &comm3d);

    // 5. Get my rank and coords in comm3d
    int rank3d;
    MPI_Comm_rank(comm3d, &rank3d);
    int coords[3];
    MPI_Cart_coords(comm3d, rank3d, 3, coords);
    int i = coords[0], j = coords[1], k = coords[2];

    // 6. Allocate local tiles: tile_size x tile_size
    int tile_size = N / p_side;
    double* A_tile = new double[tile_size * tile_size];
    double* B_tile = new double[tile_size * tile_size];
    double* C_tile = new double[tile_size * tile_size];
    // Initialize C_tile to zero
    for (int idx = 0; idx < tile_size * tile_size; idx++)
        C_tile[idx] = 0.0;

    // 7. Generate random tiles on front face (k==0)
    if (k == 0) {
        srand48(time(NULL) * (i * p_side + j + 1));
        for (int idx = 0; idx < tile_size * tile_size; idx++) {
            A_tile[idx] = drand48();
            B_tile[idx] = drand48();
        }
    }

    // 8. Create sub-communicator along k-dimension for each (i,j)
    int remain_dims[3] = {0, 0, 1};
    MPI_Comm comm_k;
    MPI_Cart_sub(comm3d, remain_dims, &comm_k);

    // 9. Broadcast initial tiles along k
    int root_k = 0; // in comm_k, rank 0 corresponds to k==0 layer
    MPI_Bcast(A_tile, tile_size * tile_size, MPI_DOUBLE, root_k, comm_k);
    MPI_Bcast(B_tile, tile_size * tile_size, MPI_DOUBLE, root_k, comm_k);

    // Synchronize before timing
    MPI_Barrier(comm3d);
    double t_start = MPI_Wtime();

    // 10. 2D shift-and-multiply loop
    for (int t = 0; t < p_side; t++) {
        // Shift A left in j-dimension
        int srcA, dstA;
        MPI_Cart_shift(comm3d, /*dim=*/1, /*disp=*/-1, &srcA, &dstA);
        MPI_Sendrecv_replace(A_tile, tile_size * tile_size, MPI_DOUBLE,
                             dstA, 0, srcA, 0, comm3d, MPI_STATUS_IGNORE);

        // Shift B up in i-dimension
        int srcB, dstB;
        MPI_Cart_shift(comm3d, /*dim=*/0, /*disp=*/-1, &srcB, &dstB);
        MPI_Sendrecv_replace(B_tile, tile_size * tile_size, MPI_DOUBLE,
                             dstB, 0, srcB, 0, comm3d, MPI_STATUS_IGNORE);

        // Local multiply: C_tile += A_tile * B_tile
        for (int ii = 0; ii < tile_size; ii++) {
            for (int jj = 0; jj < tile_size; jj++) {
                double sum = 0.0;
                for (int kk = 0; kk < tile_size; kk++) {
                    sum += A_tile[ii * tile_size + kk] * B_tile[kk * tile_size + jj];
                }
                C_tile[ii * tile_size + jj] += sum;
            }
        }
    }

    // 11. Reduce partial C tiles along k-dimension to root_k
    MPI_Reduce(C_tile, C_tile, tile_size * tile_size,
               MPI_DOUBLE, MPI_SUM, root_k, comm_k);

    // Synchronize and stop timing
    MPI_Barrier(comm3d);
    double t_end = MPI_Wtime();

    // 12. Print timing from the (0,0,0) process only
    if (i == 0 && j == 0 && k == 0) {
        printf("2.5D MatMul: N=%d, P=%d, c=%d, time=%.6f sec\n",
               N, P, c, t_end - t_start);
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
