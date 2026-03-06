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
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

namespace traccc::cuda {
  /// Base class for multiple implementations of a lamdba delegator.
  ///
  /// Intended use is to delegate a lambda to be executed in a single thread.
  /// This base class doubles as the null delegator, which simply executes
  /// the lambda in the caller thread.
  class single_threaded_delegator {
    public:
    virtual void delegate(std::function<void()> func)  {
      func();
    }
    virtual ~single_threaded_delegator() = default;
    static single_threaded_delegator& get() {
      static single_threaded_delegator instance;
      return instance;
    }
  };


  /// Derived class of single_threaded_delegator that does not propagate exceptions
  /// back to the caller. The caller does not block, and the delegated task is executed asynchronously.
  /// Exceptions handling is left to the TBB task scheduler, which will catch and log them (to be tested).
  class single_threaded_delegator_fire_and_forget : public single_threaded_delegator {
    public:
    void delegate(std::function<void()> func) override {
      m_arena.enqueue([this, func](){
          m_group.run(func);
      });
    }

    ~single_threaded_delegator_fire_and_forget() noexcept override {
      m_group.wait();
    }

    single_threaded_delegator_fire_and_forget() : m_arena(1) {}
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
  class single_threaded_delegator_sync : public single_threaded_delegator {
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

    single_threaded_delegator_sync() : m_arena(1) {}

    private:
    tbb::task_arena m_arena;
    tbb::task_group m_group;
  };
}