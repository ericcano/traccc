/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <atomic>
#include <exception>
#include <functional>
#include <iostream>
#include <syncstream>
#include <optional>
#include <thread>
#include <tbb/concurrent_queue.h>
#include <tbb/task.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

/// Define THREAD_DELEGATOR_VERBOSE before including this header to enable
/// per-call diagnostic printouts.
// #define THREAD_DELEGATOR_VERBOSE

namespace traccc::cuda {

#ifdef THREAD_DELEGATOR_VERBOSE
  /// Returns a per-call osyncstream wrapping std::cout. All instances sharing
  /// the same underlying streambuf coordinate atomically at destruction, so
  /// each chained << expression produces interleaving-free output.
  inline std::osyncstream syncout() { return std::osyncstream(std::cout); }
#endif
  /// Base class for multiple implementations of a lamdba delegator.
  ///
  /// Intended use is to delegate a lambda to be executed in a single thread.
  /// This base class doubles as the null delegator, which simply executes
  /// the lambda in the caller thread.
  class thread_delegator {
    public:
    /// Delegate a function to be executed in a single thread (or immediately, depending on the strategy). 
    /// The caller may block or suspend until the function finishes, depending on the strategy.
    virtual void delegate(std::function<void()> func)  {
#ifdef THREAD_DELEGATOR_VERBOSE
      const unsigned long serial = m_serial.fetch_add(1, std::memory_order_relaxed);
      syncout() << "[null delegator #" << serial << "] executing lambda in caller thread "
                << std::this_thread::get_id() << "\n";
#endif
      func();
#ifdef THREAD_DELEGATOR_VERBOSE
      syncout() << "[null delegator #" << serial << "] finished executing lambda in caller thread "
                << std::this_thread::get_id() << "\n";
#endif
    }
    /// Delegate a function to be executed in a single thread (or immediately, depending on the strategy). 
    /// The caller does not block. This allow implementations of delegated callback launches during
    /// await() calls.
    virtual void delegateAsync(std::function<void()> func) {
#ifdef THREAD_DELEGATOR_VERBOSE
      const unsigned long serial = m_serial.fetch_add(1, std::memory_order_relaxed);
      syncout() << "[null delegator #" << serial << "] executing async lambda in caller thread "
                << std::this_thread::get_id() << "\n";
#endif
      func();
#ifdef THREAD_DELEGATOR_VERBOSE
      syncout() << "[null delegator #" << serial << "] finished executing async lambda in caller thread "
                << std::this_thread::get_id() << "\n";
#endif
    }
    // Wait for all delegated tasks to complete. Useful for tests not calling destructors. 
    virtual void wait() {}
    virtual ~thread_delegator() = default;
    static thread_delegator& get() {
      static thread_delegator instance;
      return instance;
    }
#ifdef THREAD_DELEGATOR_VERBOSE
    std::atomic<unsigned long> m_serial{0};
#endif
  };

  /// Synchronized variant that uses TBB task suspension instead of spinlocking.
  ///
  /// The calling thread must be executing inside a TBB task context (e.g. inside
  /// a tbb::task_arena or tbb::task_group). It is suspended via
  /// tbb::this_task::suspend and woken up by the worker task once it completes.
  /// Exceptions thrown by the lambda are captured and rethrown at the call site.
  ///
  class tbb_arena_delegator_suspend : public thread_delegator {
    public:
    void delegate(std::function<void()> func) override {
      std::exception_ptr eptr = nullptr;
      const unsigned long serial = 
#ifdef THREAD_DELEGATOR_VERBOSE
          m_serial.fetch_add(1, std::memory_order_relaxed);
#else
          0;
#endif

#ifdef THREAD_DELEGATOR_VERBOSE
      syncout() << "[tbb arena delegator #" << serial << "] outer before suspend"
                << " thread=" << std::this_thread::get_id() << "\n";
#endif

      tbb::task::suspend([&](tbb::task::suspend_point tag) {
#ifdef THREAD_DELEGATOR_VERBOSE
        syncout() << "[tbb arena delegator #" << serial << "] outer lambda, about to enqueue"
                  << " thread=" << std::this_thread::get_id() << "\n";
#endif
        m_arena.enqueue([this, &func, &eptr, tag, serial]() {
          m_group.run([&func, &eptr, tag, serial]() {
#ifdef THREAD_DELEGATOR_VERBOSE
            syncout() << "[tbb arena delegator #" << serial << "] inner begin"
                      << " thread=" << std::this_thread::get_id() << "\n";
#endif
            try {
              func();
            } catch (...) {
              eptr = std::current_exception();
            }
#ifdef THREAD_DELEGATOR_VERBOSE
            syncout() << "[tbb arena delegator #" << serial << "] inner end"
                      << " thread=" << std::this_thread::get_id() << "\n";
#endif
            tbb::task::resume(tag);
          });
        });
#ifdef THREAD_DELEGATOR_VERBOSE
        syncout() << "[tbb arena delegator #" << serial << "] outer lambda, done enqueuing"
                  << " thread=" << std::this_thread::get_id() << "\n";
#endif

      });

#ifdef THREAD_DELEGATOR_VERBOSE
      syncout() << "[tbb arena delegator #" << serial << "] outer after resume"
                << " thread=" << std::this_thread::get_id() << "\n";
#endif

      if (eptr) {
        std::rethrow_exception(eptr);
      }
    }

    // The async variant swallows exceptions.
    void delegateAsync(std::function<void()> func) override {
      m_arena.enqueue([this, func](){
          m_group.run([func]() {
              try {
                  func();
#ifdef THREAD_DELEGATOR_VERBOSE
              } catch (const std::exception& e) {
                  std::cerr << "Exception in async delegated function: " << e.what() << "\n";
#endif
              } catch (...) {
#ifdef THREAD_DELEGATOR_VERBOSE
                  std::cerr << "Unknown exception in async delegated function\n";
#endif
              }
          });
      });
    }

    void wait() override {
      m_group.wait();
    }

    ~tbb_arena_delegator_suspend() noexcept override {
      m_group.wait();
    }

    static tbb_arena_delegator_suspend& get() {
      static tbb_arena_delegator_suspend instance;
#ifdef THREAD_DELEGATOR_VERBOSE
      syncout() << "Getting single_threaded_delegator_suspend instance at address " << &instance << " in thread "
                << std::this_thread::get_id() << "\n";
#endif
      return instance;
    }

    tbb_arena_delegator_suspend() : m_arena(1, 1, tbb::task_arena::priority::high) {}

    private:
    tbb::task_arena m_arena;
    tbb::task_group m_group;
#ifdef THREAD_DELEGATOR_VERBOSE
    std::atomic<unsigned long> m_serial{0};
#endif
  };




  /// Thread delegator that dispatches work to a single dedicated system thread.
  ///
  /// Unlike tbb_arena_delegator_suspend (which uses a TBB-managed arena thread),
  /// this class owns a std::thread that persists for its lifetime. The calling
  /// TBB task is suspended via tbb::task::suspend and resumed by the worker
  /// thread once the delegated work completes. Exceptions thrown by the lambda
  /// are captured and rethrown at the call site.
  class thread_delegator_suspend : public thread_delegator {
   public:
    void delegate(std::function<void()> func) override {
      std::exception_ptr eptr;
      const unsigned long serial =
#ifdef THREAD_DELEGATOR_VERBOSE
          m_serial.fetch_add(1, std::memory_order_relaxed);
#else
          0;
#endif

#ifdef THREAD_DELEGATOR_VERBOSE
      syncout() << "[sys thread delegator #" << serial << "] before suspend"
                << " thread=" << std::this_thread::get_id() << "\n";
#endif

      tbb::task::suspend([&](tbb::task::suspend_point tag) {
#ifdef THREAD_DELEGATOR_VERBOSE
        syncout() << "[sys thread delegator #" << serial
                  << "] in suspend lambda, enqueueing"
                  << " thread=" << std::this_thread::get_id() << "\n";
#endif
        m_queue.push(
            {[&func, &eptr, serial]() {
#ifdef THREAD_DELEGATOR_VERBOSE
               syncout() << "[sys thread delegator #" << serial
                         << "] worker executing"
                         << " thread=" << std::this_thread::get_id() << "\n";
#endif
               try {
                 func();
               } catch (...) {
                 eptr = std::current_exception();
               }
#ifdef THREAD_DELEGATOR_VERBOSE
               syncout() << "[sys thread delegator #" << serial
                         << "] worker done"
                         << " thread=" << std::this_thread::get_id() << "\n";
#endif
             },
             tag});
#ifdef THREAD_DELEGATOR_VERBOSE
        syncout() << "[sys thread delegator #" << serial
                  << "] enqueued, will resume via TBB"
                  << " thread=" << std::this_thread::get_id() << "\n";
#endif
      });

#ifdef THREAD_DELEGATOR_VERBOSE
      syncout() << "[sys thread delegator #" << serial << "] after resume"
                << " thread=" << std::this_thread::get_id() << "\n";
#endif

      if (eptr) {
        std::rethrow_exception(eptr);
      }
    }

    void delegateAsync(std::function<void()> func) override {
      m_queue.push({std::move(func), std::nullopt});
    }

    thread_delegator_suspend()
        : m_thread(&thread_delegator_suspend::workerLoop, this) {}

    void wait() override {
      // Wait until the queue is empty. This is not perfect since new tasks could be enqueued after we check, but it's sufficient for our current tests and avoids the complexity of tracking in-flight tasks.
      while (!m_queue.empty()) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
      }
    }

    ~thread_delegator_suspend() noexcept override {
#ifdef THREAD_DELEGATOR_VERBOSE
      syncout() << "Sending stop signal to worker thread."
                << " thread=" << std::this_thread::get_id() << "\n";
#endif
      m_queue.push({.stop_signal = true});
      if (m_thread.joinable()) {
        m_thread.join();
      } else {
        std::cerr << "Warning: thread_delegator_suspend destructor called but worker thread is not joinable\n";
      }
    }

    static thread_delegator_suspend& get() {
      static thread_delegator_suspend instance;
#ifdef THREAD_DELEGATOR_VERBOSE
      syncout() << "Getting thread_delegator_suspend instance at address "
                << &instance << " in thread " << std::this_thread::get_id()
                << "\n";
#endif
      return instance;
    }

   private:
    struct WorkItem {
      std::function<void()> func;
      std::optional<tbb::task::suspend_point> tag;
      bool stop_signal = false;
    };

    /// Worker loop: mirrors Scheduler::processRunQueue(), draining the
    /// concurrent queue with try_pop + yield.
    void workerLoop() {
      while (true) {
        WorkItem item;
        if (m_queue.try_pop(item)) {
          if (item.stop_signal) {
#ifdef THREAD_DELEGATOR_VERBOSE
            syncout() << "Worker thread received stop signal, exiting."
                      << " thread=" << std::this_thread::get_id() << "\n";
#endif
            break;
          }
          try {
            item.func();
#ifdef THREAD_DELEGATOR_VERBOSE
          } catch (const std::exception& e) {
            std::cerr << "Exception in async delegated function: " << e.what() << "\n";
#endif
          } catch (...) {
#ifdef THREAD_DELEGATOR_VERBOSE
            std::cerr << "Unknown exception in async delegated function\n";
#endif
          }
          if (item.tag) {
            tbb::task::resume(*item.tag);
          }
        } else {
          std::this_thread::yield();
        }
      }
    }

    tbb::concurrent_queue<WorkItem> m_queue;
    std::thread m_thread;
#ifdef THREAD_DELEGATOR_VERBOSE
    std::atomic<unsigned long> m_serial{0};
#endif
  };

}