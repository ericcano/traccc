/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/device/algorithm_base.hpp"
#include "traccc/utils/memory_resource.hpp"

// Local include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/cuda/utils/thread_delegator.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::cuda {

/// Base class for all CUDA algorithms
///
/// Holding on to data that all CUDA algorithms make use of.
/// Inherits from @c traccc::device::algorithm_base so that memory resources
/// and copy objects are available to the common device algorithm layer.
///
class algorithm_base : public device::algorithm_base {

    public:
    /// Constructor for the algorithm base
    ///
    /// @param mr         The memory resource(s) to use
    /// @param copy       The copy object to use
    /// @param str        The CUDA stream to perform all operations on
    /// @param delegator  The thread delegator to use for delegating tasks to a single thread
    ///
    explicit algorithm_base(const traccc::memory_resource& mr,
                            vecmem::copy& copy, cuda::stream& str,
                            thread_delegator& delegator);

    /// Get the CUDA stream of the algorithm
    cuda::stream& stream() const;
    /// Get the thread delegator of the algorithm
    thread_delegator& delegator() const;
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
