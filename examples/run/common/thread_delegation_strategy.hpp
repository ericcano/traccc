#pragma once

namespace traccc {
/// Enumeration of thread delegation strategies for CUDA runtime calls

enum class thread_delegation_strategy {
    immediate,        ///< No delegation, all code is executed in the caller thread (default)
    tbb_delegation,   ///< Delegation to a TBB single thread arena with TBB task suspension until completion
    thread_delegation ///< Delegation to a single thread TBB task suspension until completion
};

}  // namespace traccc
