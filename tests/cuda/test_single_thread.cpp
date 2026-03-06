/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "traccc/cuda/utils/single_thread_delegation.hpp"

TEST(CUDASingleThreadedDelegator, ExceptionPropagation) {
    traccc::cuda::single_threaded_delegator_sync delegator;

    EXPECT_THROW(delegator.delegate([]() {
        throw std::runtime_error("Test exception");
    }), std::runtime_error);

    // Also test that normal execution still works
    EXPECT_NO_THROW(delegator.delegate([]() {
        // Do nothing
    }));

    // Test the fire and forget delegator does not propagate exceptions
    // (actually something somewhere in TBB should catch and log the exception,
    // but we can't test that here)
    traccc::cuda::single_threaded_delegator_fire_and_forget delegator_ff;
    EXPECT_NO_THROW(delegator_ff.delegate([]() {
        throw std::runtime_error("Test exception");
    }));
}