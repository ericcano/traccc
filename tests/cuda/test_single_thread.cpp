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

//#define VERBOSE_TESTS

#ifdef VERBOSE_TESTS
#  define TEST_LOG(expr) std::osyncstream(std::cout) << expr << " from " << __FILE__ << ":" << __LINE__ << std::endl; std::this_thread::sleep_for(std::chrono::milliseconds(10))
#else
#  define TEST_LOG(expr) do {} while (false)
#endif

TEST(CUDASingleThreadedDelegator, ExceptionPropagation) {
    // Base delegator should always propagate exceptions thrown by the delegated function.
    auto &delegator = traccc::cuda::thread_delegator::get();

    for (int i = 0; i < 500; ++i) {
        EXPECT_THROW(delegator.delegate([]() {
            TEST_LOG("Throwing exception in delegate() from thread " << std::this_thread::get_id());
            throw std::runtime_error("Test exception");
        }), std::runtime_error);
    }
    // Also test that normal execution still works
    for (int i = 0; i < 500; ++i) {
        EXPECT_NO_THROW(delegator.delegate([]() {
            TEST_LOG("Running normal delegate() from thread " << std::this_thread::get_id());
            // Do nothing
        }));
    }
    // Test that exceptions thrown in the async variant do propagate to the caller
    for (int i = 0; i < 500; ++i) {
        EXPECT_THROW(delegator.delegateAsync([]() {
            TEST_LOG("Throwing exception in delegateAsync() from thread " << std::this_thread::get_id());
            throw std::runtime_error("Test exception");
        }), std::runtime_error);
    }
    delegator.wait();

    // Test TBB delegator
    auto& delegator_tbb = traccc::cuda::tbb_arena_delegator_suspend::get();
    // Exceptions thrown in the delegate() call should propagate to the caller
    for (int i = 0; i < 500; ++i) {
        EXPECT_THROW(delegator_tbb.delegate([]() {
            TEST_LOG("Throwing exception in TBB delegator delegate() from thread " << std::this_thread::get_id());
            throw std::runtime_error("Test exception");
        }), std::runtime_error);
    }
    // Also test that normal execution still works
    for (int i = 0; i < 500; ++i) {
        EXPECT_NO_THROW(delegator_tbb.delegate([]() {
            TEST_LOG("Running normal TBB delegator delegate() from thread " << std::this_thread::get_id());
            // Do nothing
        }));
     }

     // Test that exceptions are not propagated with the async variant.
     for (int i = 0; i < 500; ++i) {
         EXPECT_NO_THROW(delegator_tbb.delegateAsync([]() {
             TEST_LOG("Throwing exception in TBB delegator delegateAsync() from thread " << std::this_thread::get_id());
             throw std::runtime_error("Test exception");
         }));
     }
     delegator_tbb.wait();

     // Test thread delegator
     auto& delegator_thread = traccc::cuda::thread_delegator_suspend::get();
     // Exceptions thrown in the delegate() call should propagate to the caller
     for (int i = 0; i < 500; ++i) {
         EXPECT_THROW(delegator_thread.delegate([]() {
             TEST_LOG("Throwing exception in thread delegator delegate() from thread " << std::this_thread::get_id());
             throw std::runtime_error("Test exception");
         }), std::runtime_error);
    }
    // Also test that normal execution still works
    for (int i = 0; i < 500; ++i) {
        EXPECT_NO_THROW(delegator_thread.delegate([]() {
            TEST_LOG("Running normal thread delegator delegate() from thread " << std::this_thread::get_id());
            // Do nothing
        }));
    }

    // Test that exceptions are not propagated with the async variant.
    for (int i = 0; i < 500; ++i) {
        EXPECT_NO_THROW(delegator_thread.delegateAsync([]() {
            throw std::runtime_error("Test exception");
        }));
    }
    delegator_thread.wait();
}

TEST(CUDASingleThreadedDelegator, MultipleDelegations) {
    auto& delegator = traccc::cuda::thread_delegator::get();

    // Delegate multiple tasks and ensure they all execute correctly
    for (int i = 0; i < 500; ++i) {
        TEST_LOG("Delegating task " << i << " in thread " << std::this_thread::get_id());
        [[maybe_unused]]  volatile auto& observable_delegator = delegator; // volatile to prevent optimizations for debugging
        EXPECT_NO_THROW(delegator.delegate([i]() {
            // Just print the index to ensure the task runs
            TEST_LOG("Running task " << i << " in thread " << std::this_thread::get_id());
        }));
        TEST_LOG("Finished delegating task " << i << " in thread " << std::this_thread::get_id());
    }
    delegator.wait();

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
    delegator_suspend.wait();
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
    delegator.wait();

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
    tg.wait();
    delegator_thread.wait();

    // Same with TBB delegator
    auto& delegator_tbb = traccc::cuda::tbb_arena_delegator_suspend::get();
    for (int i = 0; i < 500; ++i) {
        outer_arena.execute([&]() {
            tg.run([i, &delegator_tbb]() {
                TEST_LOG("Outer sync task " << i << " in thread " << std::this_thread::get_id());
                EXPECT_NO_THROW(delegator_tbb.delegate([i]() {
                    TEST_LOG("Inner sync task " << i << " in thread " << std::this_thread::get_id());
                }));
            });
        });
    }
    tg.wait();
    delegator_tbb.wait();
}
