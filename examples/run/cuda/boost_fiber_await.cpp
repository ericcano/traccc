// Local include(s).
#include "boost_fiber_await.hpp"

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>  // boost/fiber/cuda/waitfor.hpp includes by mistake driver header so this must be placed before as a workaround

// Boost include(s).
#include <boost/fiber/cuda/waitfor.hpp>

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

namespace traccc::cuda {

void boost_fiber_await(const traccc::cuda::stream& stream) {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream.cudaStream());
    auto result = boost::fibers::cuda::waitfor_all(cuda_stream);
    CUDA_ERROR_CHECK(std::get<1>(result));
}

}  // namespace traccc::cuda
