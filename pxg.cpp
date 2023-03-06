#include <iostream>

#include <mpi.h>
#include <nccl.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << world << " / " << rank << std::endl;

    MPI_Finalize();
}


