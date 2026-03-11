/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <syncstream>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

#include "traccc/cuda/utils/thread_delegator.hpp"

// Define VERBOSE_TESTS to enable diagnostic printouts. TEST itself being a 
// macro, we should not have #ifdefs within the test body.

// #define VERBOSE_TESTS

#ifdef VERBOSE_TESTS
#  define TEST_LOG(expr) std::osyncstream(std::cout) << expr << std::endl
#else
#  define TEST_LOG(expr) do {} while (false)
#endif

TEST(CUDASingleThreadedDelegator, ExceptionPropagation) {
    auto &delegator = traccc::cuda::single_threaded_delegator_sync::get();

    for (int i = 0; i < 500; ++i) {
        EXPECT_THROW(delegator.delegate([]() {
            throw std::runtime_error("Test exception");
        }), std::runtime_error);
    }

    // Also test that normal execution still works
    for (int i = 0; i < 500; ++i) {
        EXPECT_NO_THROW(delegator.delegate([]() {
            // Do nothing
        }));
    }

    // Test the fire and forget delegator does not propagate exceptions
    // (actually something somewhere in TBB should catch and log the exception,
    // but we can't test that here)
    auto& delegator_ff = traccc::cuda::single_threaded_delegator_fire_and_forget::get();
    for (int i = 0; i < 500; ++i) {
         EXPECT_NO_THROW(delegator_ff.delegate([]() {
            throw std::runtime_error("Test exception");
        }));
    }

    traccc::cuda::single_threaded_delegator_suspend delegator_suspend;
    for (int i = 0; i < 500; ++i) {
        EXPECT_THROW(delegator_suspend.delegate([]() {
            throw std::runtime_error("Test exception");
        }), std::runtime_error);
    }
}

TEST(CUDASingleThreadedDelegator, MultipleDelegations) {
    auto& delegator = traccc::cuda::single_threaded_delegator_sync::get();

    // Delegate multiple tasks and ensure they all execute correctly
    for (int i = 0; i < 500; ++i) {
        TEST_LOG("Delegating task " << i << " in thread " << std::this_thread::get_id());
        EXPECT_NO_THROW(delegator.delegate([i]() {
            // Just print the index to ensure the task runs
            TEST_LOG("Running task " << i << " in thread " << std::this_thread::get_id());
        }));
        TEST_LOG("Finished delegating task " << i << " in thread " << std::this_thread::get_id());
    }

    // Delegating multiple tasks to the fire and forget delegator
    auto& delegator_ff = traccc::cuda::single_threaded_delegator_fire_and_forget::get();
    for (int i = 0; i < 500; ++i) {
        TEST_LOG("Delegating task " << i << " to fire and forget delegator in thread " << std::this_thread::get_id());
        EXPECT_NO_THROW(delegator_ff.delegate([i]() {
            TEST_LOG("Running fire and forget task " << i << " in thread " << std::this_thread::get_id());
        }));
        TEST_LOG("Finished delegating task " << i << " to fire and forget delegator in thread " << std::this_thread::get_id());
    }

    // Delegating multiple tasks to the suspend delegator
    auto&  delegator_suspend = traccc::cuda::single_threaded_delegator_suspend::get();
    for (int i = 0; i < 500; ++i) {
        TEST_LOG("Delegating task " << i << " to suspend delegator in thread " << std::this_thread::get_id());
        EXPECT_NO_THROW(delegator_suspend.delegate([i]() {
            TEST_LOG("Running suspend task " << i << " in thread " << std::this_thread::get_id());
        }));
        TEST_LOG("Finished delegating task " << i << " to suspend delegator in thread " << std::this_thread::get_id());
    }
}

TEST(CUDASingleThreadedDelegator, MultipleDelegationsInTasksSync) {
    tbb::task_arena outer_arena(std::thread::hardware_concurrency()-1);

    // Outer tasks dispatched as TBB tasks in a 10-thread arena.
    // Each outer task calls delegate() which itself enqueues an inner task.
    auto& delegator = traccc::cuda::single_threaded_delegator_sync::get();
    tbb::task_group tg;
    outer_arena.execute([&]() {
        for (int i = 0; i < 500; ++i) {
            tg.run([i, &delegator]() {
                TEST_LOG("Outer sync task " << i << " in thread " << std::this_thread::get_id());
                EXPECT_NO_THROW(delegator.delegate([i]() {
                    TEST_LOG("Inner sync task " << i << " in thread " << std::this_thread::get_id());
                }));
            });
        }
    });
    tg.wait();
}


TEST(CUDASingleThreadedDelegator, MultipleDelegationsInTasksFireAndForget) {
    tbb::task_arena outer_arena(std::thread::hardware_concurrency()-1);
    auto& delegator_ff = traccc::cuda::single_threaded_delegator_fire_and_forget::get();
    tbb::task_group tg;
    for (int i = 0; i < 500; ++i) {
        outer_arena.execute([&]() {
            tg.run([i, &delegator_ff]() {
                TEST_LOG("Outer fire-and-forget task " << i << " in thread " << std::this_thread::get_id());
                EXPECT_NO_THROW(delegator_ff.delegate([i]() {
                    TEST_LOG("Inner fire-and-forget task " << i << " in thread " << std::this_thread::get_id());
                }));
            });
        });
    }
    tg.wait();
}

TEST(CUDASingleThreadedDelegator, MultipleDelegationsInTasksSuspend) {
    tbb::task_arena outer_arena(std::thread::hardware_concurrency());
    auto& delegator_suspend = traccc::cuda::single_threaded_delegator_suspend::get();
    tbb::task_group tg;
    outer_arena.execute([&]() {
        for (int i = 0; i < 500; ++i) {
            tg.run([i, &delegator_suspend]() {
                TEST_LOG("Outer suspend task " << i << " begin in thread " << std::this_thread::get_id());
                EXPECT_NO_THROW(delegator_suspend.delegate([i]() {
                    // nano sleep to increase the chance of interleaving between tasks and make the test more robust
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    TEST_LOG("Inner suspend task " << i << " in thread " << std::this_thread::get_id());
                }));
                TEST_LOG("Outer suspend task " << i << " end in thread " << std::this_thread::get_id());
            });
        }
        tg.wait();
    });
}   