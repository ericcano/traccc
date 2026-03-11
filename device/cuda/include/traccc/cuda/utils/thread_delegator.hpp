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
#include <tbb/task.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

namespace traccc::cuda {
  /// Base class for multiple implementations of a lamdba delegator.
  ///
  /// Intended use is to delegate a lambda to be executed in a single thread.
  /// This base class doubles as the null delegator, which simply executes
  /// the lambda in the caller thread.
  class thread_delegator {
    public:
    virtual void delegate(std::function<void()> func)  {
      func();
    }
    virtual ~thread_delegator() = default;
    static thread_delegator& get() {
      static thread_delegator instance;
      return instance;
    }
  };


  /// Derived class of single_threaded_delegator that does not propagate exceptions
  /// back to the caller. The caller does not block, and the delegated task is executed asynchronously.
  /// Exceptions are discarded.
  class single_threaded_delegator_fire_and_forget : public thread_delegator {
    public:
    void delegate(std::function<void()> func) override {
      m_arena.enqueue([this, func](){
          m_group.run([this, func]() { try { func(); } catch (...) {}});
      });
    }

    ~single_threaded_delegator_fire_and_forget() noexcept override {
      try {
        m_group.wait();
      } catch (...) {
        // discard exceptions
      }
      m_group.wait();
    }

    single_threaded_delegator_fire_and_forget() : m_arena(1, 0, tbb::task_arena::priority::high) {}
    static single_threaded_delegator_fire_and_forget& get() {
      static single_threaded_delegator_fire_and_forget instance;
      return instance;
    }

    private:
    tbb::task_arena m_arena;
    tbb::task_group m_group;
  };

  /// Synchronized variant of single_threaded_delegator that propagates
  /// exceptions back to the caller.
  ///
  /// The caller blocks on a spinlock until the delegated task completes.
  /// If the lambda threw, the exception is rethrown at the call site.
  class single_threaded_delegator_sync : public thread_delegator {
    public:
    void delegate(std::function<void()> func) {
      std::exception_ptr eptr = nullptr;
      std::atomic<bool> done{false};

      m_arena.enqueue([this, &func, &eptr, &done]() {
        m_group.run([&func, &eptr, &done]() {
          try {
            func();
          } catch (...) {
            eptr = std::current_exception();
          }
          done.store(true, std::memory_order_release);
        });
      });

      while (!done.load(std::memory_order_acquire)) {
        // spinlock
      }

      if (eptr) {
        std::rethrow_exception(eptr);
      }
    }

    ~single_threaded_delegator_sync() noexcept override{
      m_group.wait();
    }
    static single_threaded_delegator_sync& get() {
      static single_threaded_delegator_sync instance;
      return instance;
    }

    single_threaded_delegator_sync() : m_arena(1, 0, tbb::task_arena::priority::high) {}

    private:
    tbb::task_arena m_arena;
    tbb::task_group m_group;
  };

  /// Synchronized variant that uses TBB task suspension instead of spinlocking.
  ///
  /// The calling thread must be executing inside a TBB task context (e.g. inside
  /// a tbb::task_arena or tbb::task_group). It is suspended via
  /// tbb::this_task::suspend and woken up by the worker task once it completes.
  /// Exceptions thrown by the lambda are captured and rethrown at the call site.
  ///
  /// Define THREAD_DELEGATOR_VERBOSE before including this header to enable
  /// per-call diagnostic printouts.
// #define THREAD_DELEGATOR_VERBOSE
  class single_threaded_delegator_suspend : public thread_delegator {
    public:
    void delegate(std::function<void()> func) override {
      std::exception_ptr eptr = nullptr;
      const unsigned long serial = m_serial.fetch_add(1, std::memory_order_relaxed);

#ifdef THREAD_DELEGATOR_VERBOSE
      std::cout << "[suspend delegator #" << serial << "] outer before suspend"
                << " thread=" << std::this_thread::get_id() << "\n";
#endif

      tbb::task::suspend([&](tbb::task::suspend_point tag) {
#ifdef THREAD_DELEGATOR_VERBOSE
        std::cout << "[suspend delegator #" << serial << "] outer lambda, about to enqueue"
                  << " thread=" << std::this_thread::get_id() << "\n";
#endif
        m_arena.enqueue([this, &func, &eptr, tag, serial]() {
          m_group.run([&func, &eptr, tag, serial]() {
#ifdef THREAD_DELEGATOR_VERBOSE
            std::cout << "[suspend delegator #" << serial << "] inner begin"
                      << " thread=" << std::this_thread::get_id() << "\n";
#endif
            try {
              func();
            } catch (...) {
              eptr = std::current_exception();
            }
#ifdef THREAD_DELEGATOR_VERBOSE
            std::cout << "[suspend delegator #" << serial << "] inner end"
                      << " thread=" << std::this_thread::get_id() << "\n";
#endif
            tbb::task::resume(tag);
          });
        });
#ifdef THREAD_DELEGATOR_VERBOSE
        std::cout << "[suspend delegator #" << serial << "] outer lambda, done enqueuing"
                  << " thread=" << std::this_thread::get_id() << "\n";
#endif

      });

#ifdef THREAD_DELEGATOR_VERBOSE
      std::cout << "[suspend delegator #" << serial << "] outer after resume"
                << " thread=" << std::this_thread::get_id() << "\n";
#endif

      if (eptr) {
        std::rethrow_exception(eptr);
      }
    }

    ~single_threaded_delegator_suspend() noexcept override {
      m_group.wait();
    }

    static single_threaded_delegator_suspend& get() {
      static single_threaded_delegator_suspend instance;
#ifdef THREAD_DELEGATOR_VERBOSE
      std::cout << "Getting single_threaded_delegator_suspend instance at address " << &instance << " in thread "
                << std::this_thread::get_id() << "\n";
#endif
      return instance;
    }

    single_threaded_delegator_suspend() : m_arena(1, 1, tbb::task_arena::priority::high) {}

    private:
    tbb::task_arena m_arena;
    tbb::task_group m_group;
    std::atomic<unsigned long> m_serial{0};
  };

  
}