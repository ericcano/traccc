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
#include "traccc/alpaka/utils/queue.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <cstddef>
#include <functional>

namespace traccc::alpaka {

/// Base class for all Alpaka algorithms
///
/// Holding on to data that all Alpaka algorithms make use of.
/// Inherits from @c traccc::device::algorithm_base so that memory resources
/// and copy objects are available to the common device algorithm layer.
///
class algorithm_base : public device::algorithm_base {

    public:
    /// Constructor
    ///
    /// @param mr  The memory resource(s) to use
    /// @param copy The copy object to use
    /// @param q   The Alpaka queue to perform the operations in
    ///
    explicit algorithm_base(const traccc::memory_resource& mr,
                            vecmem::copy& copy, alpaka::queue& q);

    /// Get the Alpaka queue of the algorithm
    alpaka::queue& queue() const;

    /// Get the preferred warp size of the device being used
    unsigned int warp_size() const;

    private:
    /// The Alpaka queue to use
    std::reference_wrapper<alpaka::queue> m_queue;
    /// Preferred warp size of the device being used
    unsigned int m_warp_size;

};  // class algorithm_base

}  // namespace traccc::alpaka
