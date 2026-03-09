/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/cuda/utils/thread_delegator.hpp"

// System include(s).
#include <functional>

namespace traccc::cuda {

/// Base class for all CUDA algorithms
///
/// Holding on to data that all CUDA algorithms make use of.
///
class algorithm_base {

    public:
    /// Constructor for the algorithm base
    ///
    /// @param str The CUDA stream to perform all operations on
    /// @param delegator The thread delegator to use for delegating tasks to a single thread
    ///
    explicit algorithm_base(cuda::stream& str, thread_delegator& delegator);

    /// Get the CUDA stream of the algorithm
    cuda::stream& stream() const;
    /// Get the warp size of the GPU being used
    unsigned int warp_size() const;

    private:
    /// The CUDA stream to use
    std::reference_wrapper<cuda::stream> m_stream;
    /// The thread delegator to use
    std::reference_wrapper<thread_delegator> m_delegator;
    /// Warp size of the GPU being used
    unsigned int m_warp_size;

};  // class algorithm_base

using await_function_t = void (*)(const cuda::stream&);

void default_await_function(const cuda::stream& stream);

}  // namespace traccc::cuda
