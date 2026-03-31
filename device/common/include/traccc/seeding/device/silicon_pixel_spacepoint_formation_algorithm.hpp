/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/algorithm_base.hpp"

// Project include(s).
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/geometry/detector_buffer.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// System include(s).
#include <atomic>
#include <iostream>
#include <type_traits>

namespace traccc::device {

/// Algorithm forming space points out of measurements
///
/// This algorithm performs the local-to-global transformation of the 2D (pixel)
/// measurements made on every detector module, into 3D spacepoint coordinates.
///
/// @tparam Base Platform-specific algorithm base class. Must inherit from
///              @c traccc::device::algorithm_base (providing mr() and copy()).
///
template <class Base>
class silicon_pixel_spacepoint_formation_algorithm
    : public algorithm<edm::spacepoint_collection::buffer(
          const detector_buffer&,
          const edm::measurement_collection<default_algebra>::const_view&)>,
      public messaging,
      public Base {

    static_assert(std::is_base_of_v<algorithm_base, Base>,
                  "Base must inherit from traccc::device::algorithm_base");

    public:
    /// Constructor for spacepoint_formation algorithm
    ///
    /// @param base   The fully constructed platform base object
    /// @param logger The logger instance to use
    ///
    silicon_pixel_spacepoint_formation_algorithm(
        Base&& base,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone())
        : messaging(std::move(logger)), Base(std::move(base)) {}

    /// Construct spacepoints from 2D silicon pixel measurements
    ///
    /// @param det Detector object
    /// @param measurements A collection of measurements
    /// @return A spacepoint buffer, with one spacepoint for every
    ///         silicon pixel measurement
    ///
    output_type operator()(
        const detector_buffer& det,
        const edm::measurement_collection<default_algebra>::const_view&
            measurements) const override {

        static std::atomic<int> s_call_id{0};
        const int call_id = s_call_id.fetch_add(1);
        std::cout << "[silicon_pixel_spacepoint_formation_algorithm #" << call_id << "] operator() start" << std::endl;

        // Get the number of measurements. In an asynchronous way if possible.
        edm::measurement_collection<default_algebra>::const_view::size_type
            n_measurements = 0u;
        if (this->mr().host) {
            vecmem::async_size size =
                this->copy().get_size(measurements, *(this->mr().host));
            // Here we could give control back to the caller, once our code allows
            // for it. (coroutines...)<-WIP
            std::cout << "[silicon_pixel_spacepoint_formation_algorithm #" << call_id << "] before await (count measurements)" << std::endl;
            this->await();
            n_measurements = size.get();
            std::cout << "[silicon_pixel_spacepoint_formation_algorithm #" << call_id << "] after await (count measurements), n_measurements=" << n_measurements << std::endl;
        } else {
            n_measurements = this->copy().get_size(measurements);
            std::cout << "[silicon_pixel_spacepoint_formation_algorithm #" << call_id << "] n_measurements=" << n_measurements << " (sync)" << std::endl;
        }

        // If there are no measurements, return right away.
        if (n_measurements == 0) {
            std::cout << "[silicon_pixel_spacepoint_formation_algorithm #" << call_id << "] no measurements, returning early" << std::endl;
            return {};
        }

        // Create the result buffer.
        edm::spacepoint_collection::buffer spacepoints(
            n_measurements, this->mr().main,
            vecmem::data::buffer_type::resizable);
        this->copy().setup(spacepoints)->ignore();

        // Launch the spacepoint formation kernel.
        std::cout << "[silicon_pixel_spacepoint_formation_algorithm #" << call_id << "] launching form_spacepoints_kernel" << std::endl;
        form_spacepoints_kernel(
            {n_measurements, det, measurements, spacepoints});
        std::cout << "[silicon_pixel_spacepoint_formation_algorithm #" << call_id << "] form_spacepoints_kernel launched, done" << std::endl;

        // Return the reconstructed spacepoints.
        return spacepoints;
    }

    protected:
    /// @name Function(s) to be implemented by derived classes
    /// @{

    /// Payload for the @c form_spacepoints_kernel function
    struct form_spacepoints_kernel_payload {
        /// The number of measurements in the event
        edm::measurement_collection<default_algebra>::const_view::size_type
            n_measurements;
        /// The detector object
        const detector_buffer& detector;
        /// The input measurements
        const edm::measurement_collection<default_algebra>::const_view&
            measurements;
        /// The output spacepoints
        edm::spacepoint_collection::view& spacepoints;
    };

    /// Launch the spacepoint formation kernel
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void form_spacepoints_kernel(
        const form_spacepoints_kernel_payload& payload) const = 0;

    /// @}

};  // class silicon_pixel_spacepoint_formation_algorithm

}  // namespace traccc::device
