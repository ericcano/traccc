/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {

/// Enumeration of await strategies for device algorithms
///
/// Controls how an architecture-specific @c algorithm_base::await() suspends
/// execution until all asynchronous device operations have completed.
///
enum class await_strategy {
    sync,               ///< Synchronous waiting (e.g. cudaStreamSynchronize)
    boost_fiber_await,  ///< Suspend the current Boost.Fiber
    tbb_await           ///< Suspend the current TBB task
};

}  // namespace traccc
