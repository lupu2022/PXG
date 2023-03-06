#include <iostream>

#include <mpi.h>
#include <nccl.h>

#include "common.hpp"
#include "config.hpp"
#include "embedding.hpp"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    pxg_assert( world == 4, "This is a static setup with world = 4!");
    if ( rank == 0 ) {
        InputEmbedding* in = new InputEmbedding();
        in->run(rank);
        delete in;
    } else if ( rank == 1) {

    } else if ( rank == 2) {

    } else if ( rank == 4) {

    } else {
        pxg_panic("Can't be here!");
    }

    MPI_Finalize();
}


