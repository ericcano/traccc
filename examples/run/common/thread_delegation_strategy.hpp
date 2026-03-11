#pragma once

namespace traccc {
/// Enumeration of thread delegation strategies for CUDA runtime calls

enum class thread_delegation_strategy {
    immediate,      ///< No delegation, all code is executed in the caller thread (default)
    fire_and_forget,///< Delegation to a single thread without waiting for completion
    sync_delegation,///< Delegation to a single thread with synchronous waiting for completion
    suspend         ///< Delegation to a single thread with TBB task suspension until completion
};

}  // namespace traccc
