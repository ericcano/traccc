#pragma once

namespace traccc {
/// Enumeration of await strategies for synchronous or suspending operations

enum class await_strategy {
    sync,               ///< Synchronous waiting
    boost_fiber_await,  ///< Suspension with Boost.Fiber
    tbb_await           ///< Suspension with TBB tasks
};

}  // namespace traccc
