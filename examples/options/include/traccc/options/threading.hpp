/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"

// System include(s).
#include <cstddef>

namespace traccc::opts {

/// Option(s) for multi-threaded code execution
class threading : public interface {

    public:
    /// @name Options
    /// @{

    enum class await_strategy {
        sync,    ///< Synchronous waiting
        suspend  ///< Suspending waiting
    };

    await_strategy await_mode = await_strategy::sync;

    enum class thread_delegation_strategy {
        immediate,      ///< No delegation, all code is executed in the caller thread (default)
        fire_and_forget,///< Delegation to a single thread without waiting for completion
        sync_delegation ///< Delegation to a single thread with synchronous waiting for completion
    };

    thread_delegation_strategy delegation_strategy = thread_delegation_strategy::immediate;

    /// The number of threads to use for the data processing
    std::size_t threads = 1;

    /// The number of events that can  be processed concurrently
    std::size_t concurrent_slots = 1;

    /// @}

    /// Constructor
    threading();

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm) override;

    std::unique_ptr<configuration_printable> as_printable() const override;
};  // struct threading

}  // namespace traccc::opts
