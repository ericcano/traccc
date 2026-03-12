// Local include(s).
#include "fiber_pool.hpp"

// Boost.Fiber include(s).
#include <boost/fiber/algo/shared_work.hpp>
#include <boost/fiber/fiber.hpp>

// System include(s).
#include <thread>

namespace traccc {
fiber_pool::fiber_pool(int nFibers) {
    m_threads.reserve(static_cast<std::size_t>(nFibers));
    m_fiber_schedulers.reserve(static_cast<std::size_t>(nFibers));
    /// initialize fiber scheduler for sharing fiber queues with workers
    boost::fibers::use_scheduling_algorithm<boost::fibers::algo::shared_work>(
        true);  /// suspend fiber scheduler on the calling thread
    for (int i = 0; i < nFibers; ++i) {
        m_threads.emplace_back([this]() {
            /// Each worker thread initializes its own fiber scheduler
            /// This code is taken from boost::fibers::use_scheduling_algorithm
            /// but stores a handle to the algorithm for later notification
            m_fiber_schedulers.emplace_back(
                new boost::fibers::algo::shared_work(
                    true));  /// `new` as in
                             /// boost::fibers::use_scheduling_algorithm
                             /// implementation
            boost::fibers::initialize_thread(
                m_fiber_schedulers.back(),
                boost::fibers::make_stack_allocator_wrapper<
                    boost::fibers::default_stack>());

            std::unique_lock<boost::fibers::mutex> lock(
                m_mutex);  /// suspend main fiber until destruction
            m_cv.wait(lock);
        });
    }
}

fiber_pool::~fiber_pool() {
    m_cv.notify_all();
}

}  // namespace traccc
