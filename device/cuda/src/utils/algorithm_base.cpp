/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/utils/algorithm_base.hpp"

#include "../utils/utils.hpp"

namespace traccc::cuda {

algorithm_base::algorithm_base(const traccc::memory_resource& mr,
                               vecmem::copy& copy, cuda::stream& str,
                               thread_delegator& delegator,
                               await_function_t await_func)
    : device::algorithm_base(mr, copy),
      m_stream(str),
      m_delegator(delegator),
      m_warp_size(details::get_warp_size(str.device())),
      m_await_function(await_func) {}

cuda::stream& algorithm_base::stream() const {

    return m_stream.get();
}

thread_delegator& algorithm_base::delegator() const {

    return m_delegator.get();
}

unsigned int algorithm_base::warp_size() const {

    return m_warp_size;
}

void default_await_function(const cuda::stream& stream) {

    stream.synchronize();
}

void algorithm_base::await() const {
    m_await_function(stream());
}

}  // namespace traccc::cuda
