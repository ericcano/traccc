#pragma once

// Boost.Fiber include(s).
#include <boost/fiber/algo/shared_work.hpp>
#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/mutex.hpp>

// System include(s).
#include <thread>
#include <vector>

namespace traccc {

/// Pool of threads executing fibers in parallel
class fiber_pool {
    public:
    /// Construct with a number of worker threads executing fibers
    fiber_pool(int n_threads);
    ~fiber_pool();
    fiber_pool(fiber_pool const&) = delete;
    fiber_pool& operator=(fiber_pool const&) = delete;
    fiber_pool(fiber_pool&&) = delete;
    fiber_pool& operator=(fiber_pool&&) = delete;

    /// Enqueue a function to be executed in the fiber pool
    template <typename T>
        requires std::invocable<T>
    void enqueue(T&& func) {
        boost::fibers::fiber(boost::fibers::launch::post, std::forward<T>(func))
            .detach();
        for (auto& scheduler : m_fiber_schedulers) {
            scheduler->notify();
        }
    }

    private:
    using fiber_algorithm_type = boost::fibers::algo::shared_work;
    std::vector<std::jthread> m_threads;  /// Worker threads
    std::vector<fiber_algorithm_type*>
        m_fiber_schedulers;        // handles to fiber scheduling algorithms
    boost::fibers::mutex m_mutex;  /// Mutex for synchronizing destruction of
                                   /// workers' main fibers
    boost::fibers::condition_variable
        m_cv;  /// Condition variable for destruction of workers' main fibers
};
}  // namespace traccc
