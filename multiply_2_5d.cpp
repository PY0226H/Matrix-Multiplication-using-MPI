// multiply_2_5d.cpp – Communication‑optimal 2.5D matrix multiplication (MPI + C++17)
// -----------------------------------------------------------------------------
// Implements Solomonik et al. 2.5‑D algorithm on a √(P/c) × √(P/c) × c processor
// grid.  Each rank owns three b×b tiles (A,B,C).
//
// Compile examples
//   # baseline   (naïve local GEMM)
//   mpicxx -O3 -std=c++17 -march=native -o multiply_2_5d multiply_2_5d.cpp
//
//   # peak flops (OpenBLAS + verification)
//   mpicxx -O3 -std=c++17 -DUSE_BLAS -lopenblas -DVERIFY -o multiply_2_5d multiply_2_5d.cpp
//
// Usage
//   mpirun -np <P> ./multiply_2_5d N [c]
//     N – global matrix dimension (square)    (required)
//     c – replication factor (optional). If omitted, code picks the
//         largest c ≤ P such that P/c is a perfect square.
//
// Flags
//   -DUSE_BLAS  : use dgemm for local multiplication (requires BLAS)
//   -DVERIFY    : for N ≤ 1024 compute serial reference and print error
//
// =============================================================================
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef USE_BLAS
extern "C" void dgemm_(const char*, const char*, const int*, const int*, const int*,
                        const double*, const double*, const int*, const double*, const int*,
                        const double*, double*, const int*);
#endif

// ----------------------------------------------------------------------------
static inline double* alloc_mat(int m, int n) {
    return static_cast<double*>(std::malloc(sizeof(double) * m * n));
}

static inline void fill_rand(double* A, int m, int n, unsigned long seed) {
    srand48(seed);
    for (long i = 0; i < (long)m * n; ++i) A[i] = drand48() - 0.5;
}

static inline void zero_mat(double* A, int m, int n) {
    std::memset(A, 0, sizeof(double) * m * n);
}

static inline void local_gemm(const double* A, const double* B, double* C, int b) {
#ifdef USE_BLAS
    const double one = 1.0, zero = 1.0;   // C ← C + A·B
    dgemm_("N", "N", &b, &b, &b, &one, A, &b, B, &b, &one, C, &b);
#else
    for (int i = 0; i < b; ++i)
        for (int k = 0; k < b; ++k) {
            double aik = A[i * b + k];
            for (int j = 0; j < b; ++j)
                C[i * b + j] += aik * B[k * b + j];
        }
#endif
}

// Choose the largest replication factor c such that P/c is a perfect square
static int choose_c(int P) {
    int best = 1;
    for (int cand = 2; cand <= P; ++cand) {
        if (P % cand) continue;
        int q = P / cand;
        int root = (int)std::round(std::sqrt(q));
        if (root * root == q) best = cand;
    }
    return best;
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) std::fprintf(stderr, "Usage: %s N [c]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int N = std::atoi(argv[1]);
    int c = (argc >= 3) ? std::atoi(argv[2]) : choose_c(size);

    if (c < 1 || size % c != 0) {
        if (rank == 0) std::fprintf(stderr, "Error: P %% c != 0 (P=%d, c=%d)\n", size, c);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int p2 = size / c;
    int p  = (int)std::round(std::sqrt(p2));
    if (p * p != p2) {
        if (rank == 0) std::fprintf(stderr, "Error: P/c (%d) is not a perfect square\n", p2);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    if (N % p != 0) {
        if (rank == 0)
            std::fprintf(stderr, "Error: N (%d) must be divisible by √(P/c) (%d)\n", N, p);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    /* ---------------- Cartesian grid ------------------------------------- */
    int dims[3]    = {p, p, c};
    int periods[3] = {1, 1, 0};          // wrap in x,y; fixed in z
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &grid);

    int coords[3];
    MPI_Cart_coords(grid, rank, 3, coords);
    int row = coords[0];
    int col = coords[1];
    int lay = coords[2];

    /* sub‑communicator along z for broadcast & reduction */
    int remain_z[3] = {0, 0, 1};
    MPI_Comm z_comm;
    MPI_Cart_sub(grid, remain_z, &z_comm);

    /* ---------------- Allocate & initialize tiles ------------------------ */
    const int b = N / p;
    double *A = alloc_mat(b, b);
    double *B = alloc_mat(b, b);
    double *C = alloc_mat(b, b);
    zero_mat(C, b, b);

    if (lay == 0) {
        fill_rand(A, b, b,  1ul * (row * p + col + 1));
        fill_rand(B, b, b, 777ul * (row * p + col + 1));
    }

    /* replicate A, B across z‑stack */
    MPI_Bcast(A, b * b, MPI_DOUBLE, 0, z_comm);
    MPI_Bcast(B, b * b, MPI_DOUBLE, 0, z_comm);

    /* ---------------- Pre‑compute shift partners ------------------------- */
    int dstA, srcA, dstB, srcB;
    MPI_Cart_shift(grid, /*dimension=*/1, /*disp=*/-1, &srcA, &dstA); // A left
    MPI_Cart_shift(grid, /*dimension=*/0, /*disp=*/-1, &srcB, &dstB); // B up

    /* ---------------- Timed multiply phase ------------------------------ */
    MPI_Barrier(grid);
    double t0 = MPI_Wtime();

    for (int step = 0; step < p; ++step) {
        local_gemm(A, B, C, b);

        // Shift A left along row
        MPI_Sendrecv_replace(A, b * b, MPI_DOUBLE, dstA, 0, srcA, 0, grid, MPI_STATUS_IGNORE);
        // Shift B up along column (tag 1 to disambiguate)
        MPI_Sendrecv_replace(B, b * b, MPI_DOUBLE, dstB, 1, srcB, 1, grid, MPI_STATUS_IGNORE);
    }

    /* reduce partial C back to front face (lay == 0) */
    if (lay == 0)
        MPI_Reduce(MPI_IN_PLACE, C, b * b, MPI_DOUBLE, MPI_SUM, 0, z_comm);
    else
        MPI_Reduce(C, nullptr,      b * b, MPI_DOUBLE, MPI_SUM, 0, z_comm);

    double t1 = MPI_Wtime();
    double local_time = t1 - t0, max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        std::printf("2.5D MM: N=%d  P=%d  c=%d  time=%.6f s  GF/s=%.2f\n",
                    N, size, c, max_time,
                    (2.0 * N * N * (double)N / 1e9) / max_time);

#ifdef VERIFY
    if (N <= 1024) {
        /* gather C tiles on root and compare to serial reference */
        const int root_rank = 0;
        std::vector<double> C_root, A_ref, B_ref, C_ref;
        if (rank == root_rank) {
            C_root.resize((size_t)N * N, 0.0);
            A_ref.resize((size_t)N * N);
            B_ref.resize((size_t)N * N);
            C_ref.resize((size_t)N * N, 0.0);
            fill_rand(A_ref.data(), N, N, 1);
            fill_rand(B_ref.data(), N, N, 777);
            /* serial reference */
            for (int i = 0; i < N; ++i)
                for (int k = 0; k < N; ++k)
                    for (int j = 0; j < N; ++j)
                        C_ref[i * N + j] += A_ref[i * N + k] * B_ref[k * N + j];
        }

        /* pack front‑face tiles into C_root */
        if (lay == 0) {
            std::vector<double> sendbuf(b * b);
            std::memcpy(sendbuf.data(), C, b * b * sizeof(double));
            MPI_Gather(sendbuf.data(), b * b, MPI_DOUBLE,
                       rank == root_rank ? C_root.data() : nullptr,
                       b * b, MPI_DOUBLE, root_rank, z_comm); // z_comm ranks correspond to same xy plane
        }

        if (rank == root_rank) {
            double err = 0.0, ref = 0.0;
            for (long idx = 0; idx < (long)N * N; ++idx) {
                double diff = C_root[idx] - C_ref[idx];
                err += diff * diff;
                ref += C_ref[idx] * C_ref[idx];
            }
            std::printf("Relative Frobenius error = %.2e\n", std::sqrt(err / ref));
        }
    }
#endif

    /* ---------------- cleanup ------------------------------------------- */
    std::free(A);
    std::free(B);
    std::free(C);
    MPI_Comm_free(&z_comm);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}
