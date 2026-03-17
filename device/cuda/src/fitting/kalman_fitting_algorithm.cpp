/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"

#include "../utils/utils.hpp"

namespace traccc::cuda {

kalman_fitting_algorithm::kalman_fitting_algorithm(
    const config_type& config, const traccc::memory_resource& mr,
    vecmem::copy& copy, cuda::stream& str, thread_delegator& delegator,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      algorithm_base(str, delegator),
      m_config{config},
      m_mr{mr},
      m_copy{copy} {}

}  // namespace traccc::cuda
