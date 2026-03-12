#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"

namespace traccc::cuda {

/// Await CUDA stream completion using TBB task suspension
void tbb_await(const traccc::cuda::stream& stream);

}  // namespace traccc::cuda
