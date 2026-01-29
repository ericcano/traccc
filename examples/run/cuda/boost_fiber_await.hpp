#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"

namespace traccc::cuda {

/// Await CUDA stream completion using Boost.Fiber
void boost_fiber_await(const traccc::cuda::stream& stream);

}  // namespace traccc::cuda
