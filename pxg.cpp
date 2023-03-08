#include <unistd.h>
#include <iostream>

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include "common.hpp"
#include "config.hpp"
#include "embedding.hpp"
#include "attention.hpp"

#include "engine/tensortype.hpp"

struct DeviceContext {
    DeviceContext(int device) : cuda_device_(device) {
        CUDACHECK( cudaSetDevice(0) );
        CUBLASCHECK( cublasCreate_v2(&cublas_handle_) );
    }
    ~DeviceContext() {
        if ( cublas_handle_ != nullptr ) {
            CUBLASCHECK( cublasDestroy_v2(cublas_handle_) );
        }
    }

    const int cuda_device_;
    cublasHandle_t cublas_handle_;
};

int main(int argc, char* argv[]) {
    int world;
    int rank;
    ncclComm_t comm;
    ncclUniqueId id;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    pxg_assert( world == 4, "This is a static setup with world = 4!");

    if ( rank == 0 ) {
        // input embedding
        InputEmbedding* in = new InputEmbedding();
        in->run(rank);
        delete in;
    } else if ( rank == 1) {
        // Attention layer0~layer15, three pass, 5 layer per pass
        ncclGetUniqueId(&id);
        std::cout << "Sending id to another! " << std::endl;
        MPI_Send(&id, sizeof(id), MPI_BYTE, 2, 0, MPI_COMM_WORLD);

        NCCLCHECK(ncclCommInitRank(&comm, 2, id, 0));
        {
            DeviceContext ctx(0);
            tt::ComputingContext::init(ctx.cuda_device_, ctx.cublas_handle_);

        }
        NCCLCHECK(ncclCommDestroy(comm));
    } else if ( rank == 2) {
        // Attention layer16~layer30, three pass, 5 layer per pass
        MPI_Recv(&id, sizeof(id), MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Received id from another!" << std::endl;

        NCCLCHECK(ncclCommInitRank(&comm, 2, id, 1));
        {
            DeviceContext ctx(1);
            tt::ComputingContext::init(ctx.cuda_device_, ctx.cublas_handle_);

        }
        NCCLCHECK(ncclCommDestroy(comm));
    } else if ( rank == 3) {
        // output embedded


    } else {
        pxg_panic("Can't be here!");
    }

    sleep(10);

    MPI_Finalize();
}


