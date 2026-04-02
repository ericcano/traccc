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
    // Base delegator should always propagate exceptions thrown by the delegated function.
    auto &delegator = traccc::cuda::thread_delegator::get();

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
    // Test that exceptions thrown in the async variant do propagate to the caller
    for (int i = 0; i < 500; ++i) {
        EXPECT_THROW(delegator.delegateAsync([]() {
            throw std::runtime_error("Test exception");
        }), std::runtime_error);
    }

    // Test TBB delegator
    auto& delegator_tbb = traccc::cuda::tbb_arena_delegator_suspend::get();
    // Exceptions thrown in the delegate() call should propagate to the caller
    for (int i = 0; i < 500; ++i) {
        EXPECT_THROW(delegator_tbb.delegate([]() {
            throw std::runtime_error("Test exception");
        }), std::runtime_error);
    }
    // Also test that normal execution still works
    for (int i = 0; i < 500; ++i) {
        EXPECT_NO_THROW(delegator_tbb.delegate([]() {
            // Do nothing
        }));
     }

     // Test that exceptions thrown in the async variant do propagate to the caller
     for (int i = 0; i < 500; ++i) {
         EXPECT_THROW(delegator_tbb.delegateAsync([]() {
             throw std::runtime_error("Test exception");
         }), std::runtime_error);
     }

     // Test thread delegator
     auto& delegator_thread = traccc::cuda::thread_delegator_suspend::get();
     // Exceptions thrown in the delegate() call should propagate to the caller
     for (int i = 0; i < 500; ++i) {
         EXPECT_THROW(delegator_thread.delegate([]() {
             throw std::runtime_error("Test exception");
         }), std::runtime_error);
    }
    // Also test that normal execution still works
    for (int i = 0; i < 500; ++i) {
        EXPECT_NO_THROW(delegator_thread.delegate([]() {
            // Do nothing
        }));
    }

    // Test that exceptions thrown in the async variant do propagate to the caller
    for (int i = 0; i < 500; ++i) {
        EXPECT_THROW(delegator_thread.delegateAsync([]() {
            throw std::runtime_error("Test exception");
        }), std::runtime_error);
    }
}

TEST(CUDASingleThreadedDelegator, MultipleDelegations) {
    auto& delegator = traccc::cuda::thread_delegator::get();

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
    auto&  delegator_suspend = traccc::cuda::tbb_arena_delegator_suspend::get();
    for (int i = 0; i < 500; ++i) {
        TEST_LOG("Delegating task " << i << " to fire and forget delegator in thread " << std::this_thread::get_id());
        EXPECT_NO_THROW(delegator_suspend.delegateAsync([i]() {
            TEST_LOG("Running fire and forget task " << i << " in thread " << std::this_thread::get_id());
        }));
        TEST_LOG("Finished delegating task " << i << " to fire and forget delegator in thread " << std::this_thread::get_id());
    }

    // Delegating multiple tasks to the suspend delegator
    for (int i = 0; i < 500; ++i) {
        TEST_LOG("Delegating task " << i << " to suspend delegator in thread " << std::this_thread::get_id());
        EXPECT_NO_THROW(delegator_suspend.delegate([i]() {
            TEST_LOG("Running suspend task " << i << " in thread " << std::this_thread::get_id());
        }));
        TEST_LOG("Finished delegating task " << i << " to suspend delegator in thread " << std::this_thread::get_id());
    }
}

TEST(CUDASingleThreadedDelegator, MultipleDelegationsInTasks) {
    tbb::task_arena outer_arena(std::thread::hardware_concurrency()-1);

    // Outer tasks dispatched as TBB tasks in a bit arena.
    // Each outer task calls delegate() which itself enqueues an inner task.
    auto& delegator = traccc::cuda::tbb_arena_delegator_suspend::get();
    tbb::task_group tg;
    for (int i = 0; i < 500; ++i) {
        outer_arena.execute([&]() {
            tg.run([i, &delegator]() {
                TEST_LOG("Outer sync task " << i << " in thread " << std::this_thread::get_id());
                EXPECT_NO_THROW(delegator.delegate([i]() {
                    TEST_LOG("Inner sync task " << i << " in thread " << std::this_thread::get_id());
                }));
            });
        });
    }
    tg.wait();

    // Same with thread delegator
    auto& delegator_thread = traccc::cuda::thread_delegator_suspend::get();
    for (int i = 0; i < 500; ++i) {
        outer_arena.execute([&]() {
            tg.run([i, &delegator_thread]() {
                TEST_LOG("Outer sync task " << i << " in thread " << std::this_thread::get_id());
                EXPECT_NO_THROW(delegator_thread.delegate([i]() {
                    TEST_LOG("Inner sync task " << i << " in thread " << std::this_thread::get_id());
                }));
            });
        });
    }
}

