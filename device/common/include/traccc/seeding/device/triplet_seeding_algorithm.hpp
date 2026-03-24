/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/algorithm_base.hpp"
#include "traccc/device/prefix_sum_element.hpp"
#include "traccc/edm/device/device_doublet.hpp"
#include "traccc/edm/device/device_triplet.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/edm/device/seeding_global_counter.hpp"
#include "traccc/edm/device/triplet_counter.hpp"
#include "traccc/seeding/spacepoint_binning_helper.hpp"

// Project include(s).
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// System include(s).
#include <cassert>
#include <memory>

namespace traccc::device {

/// Main algorithm for performing triplet track seeding
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
template <class Base>
class triplet_seeding_algorithm
    : public algorithm<edm::seed_collection::buffer(
          const edm::spacepoint_collection::const_view&)>,
      public messaging,
      public Base {

    static_assert(std::is_base_of_v<algorithm_base, Base>,
                  "Base must inherit from traccc::device::algorithm_base");

    private:
    /// Internal data type
    struct data {
        /// Configuration for the spacepoint grid forming step
        spacepoint_grid_config m_grid_config;
        /// Configuration for the seed finding step
        seedfinder_config m_finder_config;
        /// Configuration for the seed filtering step
        seedfilter_config m_filter_config;
        /// Axes for the internal spacepoint grid
        std::pair<traccc::details::spacepoint_grid_types::host::axis_p0_type,
                  traccc::details::spacepoint_grid_types::host::axis_p1_type>
            m_axes;
    };
    public:
    /// Constructor for the seed finding algorithm
    ///
    /// @param base          The fully constructed platform base object
    /// @param finder_config The seed finding configuration
    /// @param grid_config   The spacepoint grid forming configuration
    /// @param filter_config The seed filtering configuration
    /// @param logger        The logger instance to use
    ///
    triplet_seeding_algorithm(
        Base&& base, const seedfinder_config& finder_config,
        const spacepoint_grid_config& grid_config,
        const seedfilter_config& filter_config,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone())
        : messaging(std::move(logger)),
          Base(std::move(base)),
          m_data{std::make_unique<data>(data{
              grid_config, finder_config, filter_config,
              get_axes(grid_config, (this->mr().host ? *(this->mr().host)
                                                     : this->mr().main))})} {}

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints is a view of all spacepoints in the event
    /// @return the buffer of track seeds reconstructed from the spacepoints
    ///
    output_type operator()(const edm::spacepoint_collection::const_view&
                               spacepoints) const override {

        // A small sanity check.
        assert(m_data);

        // Get the number of spacepoints. In an asynchronous way if possible.
        edm::spacepoint_collection::const_view::size_type n_spacepoints = 0u;
        if (this->mr().host) {
            vecmem::async_size size =
                this->copy().get_size(spacepoints, *(this->mr().host));
            // Here we could give control back to the caller, once our code allows
            // for it. (coroutines...)<-WIP
            this->await();
            n_spacepoints = size.get();
        } else {
            n_spacepoints = this->copy().get_size(spacepoints);
        }

        if (n_spacepoints == 0) {
            return {};
        }

        // Set up the container that will be filled with the required capacities
        // for the spacepoint grid.
        const unsigned int grid_bins =
            m_data->m_axes.first.n_bins * m_data->m_axes.second.n_bins;
        vecmem::data::vector_buffer<unsigned int> grid_capacities_buffer(
            grid_bins, this->mr().main);
        this->copy().setup(grid_capacities_buffer)->ignore();
        this->copy().memset(grid_capacities_buffer, 0)->ignore();

        // Launch the grid capacity counting kernel.
        count_grid_capacities_kernel(
            {n_spacepoints, m_data->m_finder_config, m_data->m_axes.first,
             m_data->m_axes.second, spacepoints, grid_capacities_buffer});

        // Copy grid capacities back to the host.
        vecmem::vector<unsigned int> grid_capacities_host(
            this->mr().host ? this->mr().host : &(this->mr().main));
        this->copy()(grid_capacities_buffer, grid_capacities_host)->wait();

        // Create the spacepoint grid buffer and a prefix sum buffer that
        // describes it.
        traccc::details::spacepoint_grid_types::buffer grid_buffer(
            m_data->m_axes.first, m_data->m_axes.second,
            std::vector<std::size_t>(grid_capacities_host.begin(),
                                     grid_capacities_host.end()),
            this->mr().main, this->mr().host,
            vecmem::data::buffer_type::resizable);
        this->copy().setup(grid_buffer._buffer)->ignore();
        vecmem::data::vector_buffer<prefix_sum_element_t>
            grid_prefix_sum_buffer(n_spacepoints, this->mr().main,
                                   vecmem::data::buffer_type::resizable);
        this->copy().setup(grid_prefix_sum_buffer)->ignore();

        // Launch the grid population kernel.
        populate_grid_kernel({n_spacepoints, m_data->m_finder_config,
                              spacepoints, grid_buffer,
                              grid_prefix_sum_buffer});

        // Update the spacepoint counter.
        if (this->mr().host) {
            vecmem::async_size size = this->copy().get_size(
                grid_prefix_sum_buffer, *(this->mr().host));
            // Here we could give control back to the caller, once our code allows
            // for it. (coroutines...)<-WIP
            this->await();
            n_spacepoints = size.get();
        } else {
            n_spacepoints = this->copy().get_size(grid_prefix_sum_buffer);
        }

        // Set up the doublet counter buffer.
        device::doublet_counter_collection_types::buffer
            doublet_counter_buffer{n_spacepoints, this->mr().main,
                                   vecmem::data::buffer_type::resizable};
        this->copy().setup(doublet_counter_buffer)->ignore();

        // Set up a global counter used in the seeding kernels.
        vecmem::unique_alloc_ptr<device::seeding_global_counter>
            globalCounter_device =
                vecmem::make_unique_alloc<device::seeding_global_counter>(
                    this->mr().main);
        this->copy()
            .memset(vecmem::data::vector_view<device::seeding_global_counter>(
                        1u, globalCounter_device.get()),
                    0)
            ->ignore();

        // Launch the doublet counting kernel.
        count_doublets_kernel(
            {n_spacepoints, m_data->m_finder_config, spacepoints, grid_buffer,
             grid_prefix_sum_buffer, doublet_counter_buffer,
             globalCounter_device->m_nMidBot,
             globalCounter_device->m_nMidTop});

        // Get the number of doublets found.
        device::doublet_counter_collection_types::buffer::size_type
            n_doublets = 0u;
        if (this->mr().host) {
            vecmem::async_size size = this->copy().get_size(
                doublet_counter_buffer, *(this->mr().host));
            // Here we could give control back to the caller, once our code allows
            // for it. (coroutines...)<-WIP
            this->await();
            n_doublets = size.get();
        } else {
            n_doublets = this->copy().get_size(doublet_counter_buffer);
        }
        vecmem::unique_alloc_ptr<device::seeding_global_counter>
            globalCounter_host =
                vecmem::make_unique_alloc<device::seeding_global_counter>(
                    this->mr().host ? *(this->mr().host) : this->mr().main);
        this->copy()(
                 vecmem::data::vector_view<device::seeding_global_counter>(
                     1u, globalCounter_device.get()),
                 vecmem::data::vector_view<device::seeding_global_counter>(
                     1u, globalCounter_host.get()))
            ->wait();

        // Exit already here if we won't find any triplets anyway.
        if ((globalCounter_host->m_nMidBot == 0) ||
            (globalCounter_host->m_nMidTop == 0)) {
            return {};
        }

        // Set up the doublet buffers.
        device_doublet_collection_types::buffer doublet_buffer_mb{
            globalCounter_host->m_nMidBot, this->mr().main};
        this->copy().setup(doublet_buffer_mb)->ignore();
        device_doublet_collection_types::buffer doublet_buffer_mt{
            globalCounter_host->m_nMidTop, this->mr().main};
        this->copy().setup(doublet_buffer_mt)->ignore();

        // Launch the doublet finding kernel.
        find_doublets_kernel({n_doublets, m_data->m_finder_config, spacepoints,
                              grid_buffer, doublet_counter_buffer,
                              doublet_buffer_mb, doublet_buffer_mt});

        // Set up the triplet counter buffers.
        triplet_counter_spM_collection_types::buffer
            triplet_counter_spM_buffer{n_doublets, this->mr().main};
        this->copy().setup(triplet_counter_spM_buffer)->ignore();
        this->copy().memset(triplet_counter_spM_buffer, 0)->ignore();
        triplet_counter_collection_types::buffer
            triplet_counter_midBot_buffer{
                globalCounter_host->m_nMidBot, this->mr().main,
                vecmem::data::buffer_type::resizable};
        this->copy().setup(triplet_counter_midBot_buffer)->ignore();

        // Launch the triplet counting kernel.
        count_triplets_kernel(
            {globalCounter_host->m_nMidBot, m_data->m_finder_config,
             spacepoints, grid_buffer, doublet_counter_buffer,
             doublet_buffer_mb, doublet_buffer_mt, triplet_counter_spM_buffer,
             triplet_counter_midBot_buffer});

        // Launch the triplet count reduction kernel.
        triplet_counts_reduction_kernel(
            {n_doublets, doublet_counter_buffer, triplet_counter_spM_buffer,
             globalCounter_device->m_nTriplets});

        // Get the number of triplets found.
        triplet_counter_collection_types::buffer::size_type n_midBotTriplets =
            0u;
        if (this->mr().host) {
            vecmem::async_size size = this->copy().get_size(
                triplet_counter_midBot_buffer, *(this->mr().host));
            // Here we could give control back to the caller, once our code allows
            // for it. (coroutines...)<-WIP
            this->await();
            n_midBotTriplets = size.get();
        } else {
            n_midBotTriplets =
                this->copy().get_size(triplet_counter_midBot_buffer);
        }
        this->copy()(
                 vecmem::data::vector_view<device::seeding_global_counter>(
                     1u, globalCounter_device.get()),
                 vecmem::data::vector_view<device::seeding_global_counter>(
                     1u, globalCounter_host.get()))
            ->wait();

        // If no triplets could be found, exit already here.
        if (globalCounter_host->m_nTriplets == 0) {
            return {};
        }

        // Set up the triplet buffer.
        device_triplet_collection_types::buffer triplet_buffer{
            globalCounter_host->m_nTriplets, this->mr().main};
        this->copy().setup(triplet_buffer)->ignore();

        // Launch the triplet finding kernel.
        find_triplets_kernel(
            {n_midBotTriplets, m_data->m_finder_config, m_data->m_filter_config,
             spacepoints, grid_buffer, doublet_counter_buffer, doublet_buffer_mt,
             triplet_counter_spM_buffer, triplet_counter_midBot_buffer,
             triplet_buffer});

        // Launch the triplet weight updating/filling kernel.
        update_triplet_weights_kernel(
            {globalCounter_host->m_nTriplets, m_data->m_filter_config,
             spacepoints, triplet_counter_spM_buffer,
             triplet_counter_midBot_buffer, triplet_buffer});

        // Create the result object.
        edm::seed_collection::buffer seed_buffer(
            globalCounter_host->m_nTriplets, this->mr().main,
            vecmem::data::buffer_type::resizable);
        this->copy().setup(seed_buffer)->ignore();

        // Launch the seed selecting/filling kernel.
        select_seeds_kernel(
            {n_doublets, m_data->m_finder_config, m_data->m_filter_config,
             spacepoints, grid_buffer, triplet_counter_spM_buffer,
             triplet_counter_midBot_buffer, triplet_buffer, seed_buffer});

        // Return the seed buffer.
        return seed_buffer;
    }


    protected:
    /// @name Function(s) to be implemented by derived classes
    /// @{

    /// Payload for the @c count_grid_capacities_kernel function
    struct count_grid_capacities_kernel_payload {
        /// The number of spacepoints in the event
        edm::spacepoint_collection::const_view::size_type n_spacepoints;
        /// The seed finding configuration
        const seedfinder_config& config;
        /// The phi axis of the spacepoint grid
        const traccc::details::spacepoint_grid_types::host::axis_p0_type&
            phi_axis;
        /// The z axis of the spacepoint grid
        const traccc::details::spacepoint_grid_types::host::axis_p1_type&
            z_axis;
        /// All spacepoints in the event
        const edm::spacepoint_collection::const_view& spacepoints;
        /// The buffer to write the grid capacities into
        vecmem::data::vector_view<unsigned int>& grid_capacities;
    };

    /// Spacepoint grid capacity counting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void count_grid_capacities_kernel(
        const count_grid_capacities_kernel_payload& payload) const = 0;

    /// Payload for the @c populate_grid_kernel function
    struct populate_grid_kernel_payload {
        /// The number of spacepoints in the event
        edm::spacepoint_collection::const_view::size_type n_spacepoints;
        /// The seed finding configuration
        const seedfinder_config& config;
        /// All spacepoints in the event
        const edm::spacepoint_collection::const_view& spacepoints;
        /// The spacepoint grid to populate
        traccc::details::spacepoint_grid_types::view& grid;
        /// A prefix sum describing the grid contents
        vecmem::data::vector_view<prefix_sum_element_t>& grid_prefix_sum;
    };

    /// Spacepoint grid population kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void populate_grid_kernel(
        const populate_grid_kernel_payload& payload) const = 0;

    /// Payload for the @c count_doublets_kernel function
    struct count_doublets_kernel_payload {
        /// The number of spacepoints in the event
        edm::spacepoint_collection::const_view::size_type n_spacepoints;
        /// The seed finding configuration
        const seedfinder_config& config;
        /// All spacepoints in the event
        const edm::spacepoint_collection::const_view& spacepoints;
        /// The populated spacepoint grid
        const traccc::details::spacepoint_grid_types::const_view& grid;
        /// A prefix sum describing the grid contents
        const vecmem::data::vector_view<const prefix_sum_element_t>&
            grid_prefix_sum;
        /// The doublet counter collection to fill
        doublet_counter_collection_types::view& doublet_counter;
        /// The number of middle-bottom doublets found
        unsigned int& nMidBot;
        /// The number of middle-top doublets found
        unsigned int& nMidTop;
    };

    /// Doublet counting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void count_doublets_kernel(
        const count_doublets_kernel_payload& payload) const = 0;

    /// Payload for the @c find_doublets_kernel function
    struct find_doublets_kernel_payload {
        /// The number of doublets counted earlier
        device::doublet_counter_collection_types::const_view::size_type
            n_doublets;
        /// The seed finding configuration
        const seedfinder_config& config;
        /// All spacepoints in the event
        const edm::spacepoint_collection::const_view& spacepoints;
        /// The populated spacepoint grid
        const traccc::details::spacepoint_grid_types::const_view& grid;
        /// The doublet counter collection
        const doublet_counter_collection_types::const_view& doublet_counter;
        /// The middle-bottom doublet collection to fill
        device_doublet_collection_types::view& mb_doublets;
        /// The middle-top doublet collection to fill
        device_doublet_collection_types::view& mt_doublets;
    };

    /// Doublet finding kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void find_doublets_kernel(
        const find_doublets_kernel_payload& payload) const = 0;

    /// Payload for the @c count_triplets_kernel function
    struct count_triplets_kernel_payload {
        /// The number of middle-bottom doublets found earlier
        unsigned int nMidBot;
        /// The seed finding configuration
        const seedfinder_config& config;
        /// All spacepoints in the event
        const edm::spacepoint_collection::const_view& spacepoints;
        /// The populated spacepoint grid
        const traccc::details::spacepoint_grid_types::const_view& grid;
        /// The doublet counter collection
        const doublet_counter_collection_types::const_view& doublet_counter;
        /// The middle-bottom doublet collection
        const device_doublet_collection_types::const_view& mb_doublets;
        /// The middle-top doublet collection
        const device_doublet_collection_types::const_view& mt_doublets;
        /// The triplet counter per middle spacepoint to fill
        triplet_counter_spM_collection_types::view& spM_counter;
        /// The triplet counter per middle-bottom doublet to fill
        triplet_counter_collection_types::view& midBot_counter;
    };

    /// Triplet counting kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void count_triplets_kernel(
        const count_triplets_kernel_payload& payload) const = 0;

    /// Payload for the @c triplet_counts_reduction_kernel function
    struct triplet_counts_reduction_kernel_payload {
        /// The number of doublets found earlier
        device::doublet_counter_collection_types::const_view::size_type
            n_doublets;
        /// The doublet counter collection
        const doublet_counter_collection_types::const_view& doublet_counter;
        /// The triplet counter per middle spacepoint
        triplet_counter_spM_collection_types::view& spM_counter;
        /// The total number of triplets found
        unsigned int& nTriplets;
    };

    /// Triplet count reduction kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void triplet_counts_reduction_kernel(
        const triplet_counts_reduction_kernel_payload& payload) const = 0;

    /// Payload for the @c find_triplets_kernel function
    struct find_triplets_kernel_payload {
        /// The number of middle-bottom doublets found earlier
        unsigned int nMidBot;
        /// The seed finding configuration
        const seedfinder_config& finding_config;
        /// The seed filtering configuration
        const seedfilter_config& filter_config;
        /// All spacepoints in the event
        const edm::spacepoint_collection::const_view& spacepoints;
        /// The populated spacepoint grid
        const traccc::details::spacepoint_grid_types::const_view& grid;
        /// The doublet counter collection
        const doublet_counter_collection_types::const_view& doublet_counter;
        /// The middle-top doublet collection
        const device_doublet_collection_types::const_view& mt_doublets;
        /// The triplet counter per middle spacepoint
        const triplet_counter_spM_collection_types::const_view& spM_tc;
        /// The triplet counter per middle-bottom doublet
        const triplet_counter_collection_types::const_view& midBot_tc;
        /// The triplet collection to fill
        device_triplet_collection_types::view& triplets;
    };

    /// Triplet finding kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void find_triplets_kernel(
        const find_triplets_kernel_payload& payload) const = 0;

    /// Payload for the @c update_triplet_weights_kernel function
    struct update_triplet_weights_kernel_payload {
        /// The number of triplets found earlier
        device_triplet_collection_types::const_view::size_type n_triplets;
        /// The seed filtering configuration
        const seedfilter_config& config;
        /// All spacepoints in the event
        const edm::spacepoint_collection::const_view& spacepoints;
        /// The triplet counter per middle spacepoint
        const triplet_counter_spM_collection_types::const_view& spM_tc;
        /// The triplet counter per middle-bottom doublet
        const triplet_counter_collection_types::const_view& midBot_tc;
        /// The triplet collection to update
        device_triplet_collection_types::view& triplets;
    };

    /// Triplet weight updater/filler kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void update_triplet_weights_kernel(
        const update_triplet_weights_kernel_payload& payload) const = 0;

    /// Payload for the @c select_seeds_kernel function
    struct select_seeds_kernel_payload {
        /// The number of doublets found earlier
        device::doublet_counter_collection_types::const_view::size_type
            n_doublets;
        /// The seed finding configuration
        const seedfinder_config& finder_config;
        /// The seed filtering configuration
        const seedfilter_config& filter_config;
        /// All spacepoints in the event
        const edm::spacepoint_collection::const_view& spacepoints;
        /// The populated spacepoint grid
        const traccc::details::spacepoint_grid_types::const_view& grid;
        /// The triplet counter per middle spacepoint
        const triplet_counter_spM_collection_types::const_view& spM_tc;
        /// The triplet counter per middle-bottom doublet
        const triplet_counter_collection_types::const_view& midBot_tc;
        /// The triplet collection
        const device_triplet_collection_types::const_view& triplets;
        /// The seed collection to fill
        edm::seed_collection::view& seeds;
    };

    /// Seed selection/filling kernel launcher
    ///
    /// @param payload The payload for the kernel
    ///
    virtual void select_seeds_kernel(
        const select_seeds_kernel_payload& payload) const = 0;

    /// @}

    private:
    /// Pointer to internal data
    std::unique_ptr<data> m_data;

};  // class triplet_seeding_algorithm

}  // namespace traccc::device
