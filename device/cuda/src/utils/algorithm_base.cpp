/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/utils/algorithm_base.hpp"

#include "../utils/utils.hpp"

// CUDA include(s).
// boost/fiber/cuda/waitfor.hpp includes by mistake the driver header, so this
// must be placed before it as a workaround.
#include <cuda_runtime_api.h>

// Boost include(s).
#include <boost/fiber/cuda/waitfor.hpp>

// TBB include(s).
#include <tbb/task.h>

// System include(s).
#include <stdexcept>
#include <string>

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

namespace {

void tbb_await_callback(void* tag) {
    tbb::task::resume(*static_cast<tbb::task::suspend_point*>(tag));
}

}  // anonymous namespace

namespace traccc::cuda {

algorithm_base::algorithm_base(const traccc::memory_resource& mr,
                               vecmem::copy& copy, cuda::stream& str,
                               thread_delegator& delegator,
                               traccc::await_strategy strategy)
    : device::algorithm_base(mr, copy),
      m_stream(str),
      m_delegator(delegator),
      m_warp_size(details::get_warp_size(str.device())),
      m_await_strategy(strategy) {}

cuda::stream& algorithm_base::stream() const {

    return m_stream.get();
}

thread_delegator& algorithm_base::delegator() const {

    return m_delegator.get();
}

unsigned int algorithm_base::warp_size() const {

    return m_warp_size;
}

void algorithm_base::await() const {
    switch (m_await_strategy) {
        case traccc::await_strategy::sync:
            stream().synchronize();
            break;
        case traccc::await_strategy::boost_fiber_await: {
            auto s = reinterpret_cast<cudaStream_t>(stream().cudaStream());
            auto result = boost::fibers::cuda::waitfor_all(s);
            CUDA_ERROR_CHECK(std::get<1>(result));
            break;
        }
        case traccc::await_strategy::tbb_await: {
            tbb::task::suspend_point suspend_point;
            tbb::task::suspend([&suspend_point, this](auto tag) {
                suspend_point = tag;
                auto s = reinterpret_cast<cudaStream_t>(stream().cudaStream());
                delegator().delegateAsync([s, &suspend_point]() {
                    CUDA_ERROR_CHECK(
                        cudaLaunchHostFunc(s, tbb_await_callback, &suspend_point));
                    std::cout << "[tbb_await_callback] CUDA callback launched from thread "
                              << std::this_thread::get_id() << std::endl;
                });
            });
            CUDA_ERROR_CHECK(cudaGetLastError());
            break;
        }
        default:
            throw std::invalid_argument("Unknown await strategy");
    }
}

}  // namespace traccc::cuda
