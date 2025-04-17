#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>

// 2.5D Parallel Matrix Multiplication Skeleton
// ------------------------------------------------
// A: MxN, B: NxP, C: MxP with M=P=N for simplicity (square matrix)
// Uses a 3D grid of processors of size p_side x p_side x c

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // 1. Parse command-line arguments: N (matrix size), c (replication factor)
    if (argc != 3) {
        if (MPI::COMM_WORLD.Get_rank() == 0)
            fprintf(stderr, "Usage: %s <matrix_dim_N> <replication_c>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int N = atoi(argv[1]);           // global matrix dimension
    int c = atoi(argv[2]);           // number of replicas

    // 2. Get total number of ranks (P) and check validity
    int P;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    if (P % c != 0) {
        if (MPI::COMM_WORLD.Get_rank() == 0)
            fprintf(stderr, "Error: P must be divisible by c\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // 3. Compute p_side = sqrt(P/c) and check it's an integer
    int p_side = (int)std::sqrt(P / c);
    if (p_side * p_side * c != P) {
        if (MPI::COMM_WORLD.Get_rank() == 0)
            fprintf(stderr, "Error: P/c must be a perfect square\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // 4. Create a 3D Cartesian communicator: dims = [p_side, p_side, c]
    int dims[3] = {p_side, p_side, c};
    int periods[3] = {1, 1, 1};      // allow wraparound for shifts
    MPI_Comm comm3d;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, /*reorder=*/0, &comm3d);

    // 5. Determine my coordinates in the 3D grid
    int coords[3];
    MPI_Cart_coords(comm3d, MPI::COMM_WORLD.Get_rank(), 3, coords);
    int i = coords[0], j = coords[1], k = coords[2];

    // 6. Allocate local tiles: each tile is (N/p_side) x (N/p_side)
    int tile_size = N / p_side;
    double* A_tile = new double[tile_size * tile_size];
    double* B_tile = new double[tile_size * tile_size];
    double* C_tile = new double[tile_size * tile_size];
    // Initialize local C to zero
    for (int idx = 0; idx < tile_size*tile_size; idx++) C_tile[idx] = 0.0;

    // 7. On the front face (k==0), generate random tiles of A and B
    if (k == 0) {
        srand48(time(NULL) * (i*p_side + j));
        for (int idx = 0; idx < tile_size*tile_size; idx++) {
            A_tile[idx] = drand48();
            B_tile[idx] = drand48();
        }
    }

    // 8. Split communicator for broadcasts along the k-dimension for each (i,j)
    int remain_dims_k[3] = {0, 0, 1};
    MPI_Comm comm_k;
    MPI_Cart_sub(comm3d, remain_dims_k, &comm_k);
    int root_k = 0;  // rank with k==0 in each sub-communicator

    // 9. Broadcast A_tile and B_tile along k so all replicas have initial tiles
    MPI_Bcast(A_tile, tile_size*tile_size, MPI_DOUBLE, root_k, comm_k);
    MPI_Bcast(B_tile, tile_size*tile_size, MPI_DOUBLE, root_k, comm_k);

    // Barrier before timing
    MPI_Barrier(comm3d);
    double t_start = MPI_Wtime();

    // 10. Main 2D shift-and-multiply loop for t in 0..p_side-1
    for (int t = 0; t < p_side; t++) {
        // 10a. Shift A left by one in the row
        int srcA, dstA;
        MPI_Cart_shift(comm3d, /*dim=*/1, /*disp=*/-1, &srcA, &dstA);
        MPI_Sendrecv_replace(A_tile, tile_size*tile_size, MPI_DOUBLE,
                             dstA, 0, srcA, 0, comm3d, MPI_STATUS_IGNORE);

        // 10b. Shift B up by one in the column
        int srcB, dstB;
        MPI_Cart_shift(comm3d, /*dim=*/0, /*disp=*/-1, &srcB, &dstB);
        MPI_Sendrecv_replace(B_tile, tile_size*tile_size, MPI_DOUBLE,
                             dstB, 0, srcB, 0, comm3d, MPI_STATUS_IGNORE);

        // 10c. Local matrix multiply: C_tile += A_tile * B_tile
        for (int ii = 0; ii < tile_size; ii++) {
            for (int jj = 0; jj < tile_size; jj++) {
                double sum = 0.0;
                for (int kk = 0; kk < tile_size; kk++) {
                    sum += A_tile[ii*tile_size + kk] * B_tile[kk*tile_size + jj];
                }
                C_tile[ii*tile_size + jj] += sum;
            }
        }
    }

    // 11. Reduce C_tile along k-dimension to sum partial results
    MPI_Reduce(
        C_tile, /*sendbuf*/
        C_tile, /*recvbuf, valid on root only*/
        tile_size*tile_size,
        MPI_DOUBLE,
        MPI_SUM,
        root_k,
        comm_k
    );

    // Barrier and stop timing
    MPI_Barrier(comm3d);
    double t_end = MPI_Wtime();

    // 12. Print elapsed time on rank (0,0,0)
    if (i == 0 && j == 0 && k == 0) {
        printf("2.5D MatMul: N=%d, P=%d, c=%d, time=%f seconds\n",
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
