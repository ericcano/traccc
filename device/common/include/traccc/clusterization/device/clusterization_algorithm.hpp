/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/clusterization/device/tags.hpp"
#include "traccc/device/algorithm_base.hpp"

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/silicon_cluster_collection.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/memory/unique_ptr.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>
#include <optional>
#include <type_traits>

namespace traccc::device {

/// Base class for the algorithms performing hit clusterization
///
/// This algorithm implements hit clusterization in a massively-parallel
/// approach. Each thread handles a pre-determined number of detector cells.
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying the buffer.
///
/// @tparam Base Platform-specific algorithm base class. Must inherit from
///              @c traccc::device::algorithm_base (providing mr() and copy()).
///
template <class Base>
class clusterization_algorithm
    : public algorithm<edm::measurement_collection<default_algebra>::buffer(
          const edm::silicon_cell_collection::const_view&,
          const silicon_detector_description::const_view&)>,
      public algorithm<edm::measurement_collection<default_algebra>::buffer(
          const edm::silicon_cell_collection::const_view&,
          const silicon_detector_description::const_view&,
          clustering_discard_disjoint_set&&)>,
      public algorithm<
          std::pair<edm::measurement_collection<default_algebra>::buffer,
                    edm::silicon_cluster_collection::buffer>(
              const edm::silicon_cell_collection::const_view&,
              const silicon_detector_description::const_view&,
              clustering_keep_disjoint_set&&)>,
      public messaging,
      public Base {

    static_assert(std::is_base_of_v<algorithm_base, Base>,
                  "Base must inherit from traccc::device::algorithm_base");

    public:
    /// Configuration type
    using config_type = clustering_config;

    /// Constructor for clusterization algorithm
    ///
    /// @param base   The fully constructed platform base object
    /// @param config The clustering configuration
    /// @param logger The logger instance to use
    ///
    clusterization_algorithm(
        Base&& base, const config_type& config,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone())
        : messaging(std::move(logger)),
          Base(std::move(base)),
          m_config{config},
          m_f_backup{m_config.backup_size(), this->mr().main},
          m_gf_backup{m_config.backup_size(), this->mr().main},
          m_backup_mutex{vecmem::make_unique_alloc<unsigned int>(this->mr().main)},
          m_adjc_backup{m_config.backup_size(), this->mr().main},
          m_adjv_backup{m_config.backup_size() * 8, this->mr().main} {

        this->copy().setup(m_f_backup)->wait();
        this->copy().setup(m_gf_backup)->wait();
        this->copy().setup(m_adjc_backup)->wait();
        this->copy().setup(m_adjv_backup)->wait();
        this->copy()
            .memset(
                vecmem::data::vector_view<unsigned int>{1, m_backup_mutex.get()}, 0)
            ->wait();
    }

    /// Callable operator for clusterization algorithm
    /// @{
    edm::measurement_collection<default_algebra>::buffer operator()(
        const edm::silicon_cell_collection::const_view& cells,
        const silicon_detector_description::const_view& det_descr)
        const override {

        return this->operator()(cells, det_descr,
                                clustering_discard_disjoint_set{});
    }

    edm::measurement_collection<default_algebra>::buffer operator()(
        const edm::silicon_cell_collection::const_view& cells,
        const silicon_detector_description::const_view& det_descr,
        clustering_discard_disjoint_set&&) const override {

        static constexpr bool KEEP_DISJOINT_SET = false;
        auto [res, djs] = execute_impl(cells, det_descr, KEEP_DISJOINT_SET);
        assert(!djs.has_value());
        return std::move(res);
    }

    std::pair<edm::measurement_collection<default_algebra>::buffer,
              edm::silicon_cluster_collection::buffer>
    operator()(const edm::silicon_cell_collection::const_view& cells,
               const silicon_detector_description::const_view& det_descr,
               clustering_keep_disjoint_set&&) const override {

        static constexpr bool KEEP_DISJOINT_SET = true;
        auto [res, djs] = execute_impl(cells, det_descr, KEEP_DISJOINT_SET);
        assert(djs.has_value());
        return {std::move(res), std::move(*djs)};
    }
    /// @}

    protected:
    /// @name Function(s) to be implemented by derived classes
    /// @{

    /// Function meant to perform sanity checks on the input data
    ///
    /// @param cells     All cells in an event
    /// @return @c true if the input data is valid, @c false otherwise
    ///
    virtual bool input_is_valid(
        const edm::silicon_cell_collection::const_view& cells) const = 0;

    /// Payload for the @c ccl_kernel function
    struct ccl_kernel_payload {
        /// Number of cells in the event
        unsigned int n_cells;
        /// The clustering configuration
        const config_type& config;
        /// All cells in an event
        const edm::silicon_cell_collection::const_view& cells;
        /// The detector description
        const silicon_detector_description::const_view& det_descr;
        /// The measurement collection to fill
        edm::measurement_collection<default_algebra>::view& measurements;
        /// Buffer for linking cells to measurements
        vecmem::data::vector_view<unsigned int>& cell_links;
        /// Buffer for backup of the first element links
        vecmem::data::vector_view<details::index_t>& f_backup;
        /// Buffer for backup of the group first element links
        vecmem::data::vector_view<details::index_t>& gf_backup;
        /// Buffer for backup of the adjacency matrix (counts)
        vecmem::data::vector_view<unsigned char>& adjc_backup;
        /// Buffer for backup of the adjacency matrix (values)
        vecmem::data::vector_view<details::index_t>& adjv_backup;
        /// Mutex for the backup structures
        unsigned int* backup_mutex;
        /// Buffer for the disjoint set data structure
        vecmem::data::vector_view<unsigned int>& disjoint_set;
        /// Buffer for the sizes of the clusters
        vecmem::data::vector_view<unsigned int>& cluster_sizes;
    };

    /// Main CCL kernel launcher
    ///
    /// @param payload The payload containing all necessary data for the kernel
    ///
    virtual void ccl_kernel(const ccl_kernel_payload& payload) const = 0;

    /// Cluster data reification kernel launcher
    ///
    /// @param num_cells    Number of cells in the event
    /// @param disjoint_set Buffer for the disjoint set data structure
    /// @param cluster_data The cluster collection to fill
    ///
    virtual void cluster_maker_kernel(
        unsigned int num_cells,
        const vecmem::data::vector_view<unsigned int>& disjoint_set,
        edm::silicon_cluster_collection::view& cluster_data) const = 0;

    /// @}

    private:
    /// Main algorithmic implementation of the clusterization algorithm
    std::pair<edm::measurement_collection<default_algebra>::buffer,
              std::optional<edm::silicon_cluster_collection::buffer>>
    execute_impl(const edm::silicon_cell_collection::const_view& cells,
                 const silicon_detector_description::const_view& det_descr,
                 bool keep_disjoint_set) const {
        // Check the input data in debug mode.
        assert(input_is_valid(cells));

        // Get the number of cells, in an asynchronous way if possible.
        edm::silicon_cell_collection::const_view::size_type num_cells = 0u;
        if (this->mr().host) {
            const vecmem::async_size size = this->copy().get_size(cells, *(this->mr().host));
            // Potential delegation to cUDA thread.
            //
            // using copy_t = decltype(copy().get_size(cells, *(this->mr().host)));
            // constexpr std::size_t copy_size = sizeof(copy_t); 
            // std::byte size_storage[copy_size];
            // auto& size = *reinterpret_cast<copy_t*>(size_storage);
            // delegator().delegate([this, &cells, &num_cells, &size]() {
            //     new(&size) copy_t(copy().get_size(cells, *(this->mr().host)));
            // });
            
            // Here we could give control back to the caller, once our code allows
            // for it. (coroutines...)<-WIP
            this->await();
            num_cells = size.get();
        } else {
            num_cells = this->copy().get_size(cells);
        }

        // If there are no cells, return right away.
        if (num_cells == 0) {
            if (keep_disjoint_set) {
                return {edm::measurement_collection<default_algebra>::buffer{},
                        edm::silicon_cluster_collection::buffer{}};
            } else {
                return {};
            }
        }

        // Create the result object, overestimating the number of measurements.
        edm::measurement_collection<default_algebra>::buffer measurements{
            num_cells, this->mr().main, vecmem::data::buffer_type::resizable};
        this->copy().setup(measurements)->ignore();

        // Create buffer for linking cells to their measurements.
        vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, this->mr().main);
        this->copy().setup(cell_links)->ignore();

        // Ensure that the chosen maximum cell count is compatible with the maximum
        // stack size.
        assert(m_config.max_cells_per_thread <=
            device::details::CELLS_PER_THREAD_STACK_LIMIT);

        // If we are keeping the disjoint set data structure, allocate space for it.
        vecmem::data::vector_buffer<unsigned int> disjoint_set;
        vecmem::data::vector_buffer<unsigned int> cluster_sizes;
        if (keep_disjoint_set) {
            disjoint_set = {num_cells, this->mr().main};
            cluster_sizes = {num_cells, this->mr().main};
        }

        // Launch the CCL kernel.
        ccl_kernel({num_cells, m_config, cells, det_descr, measurements, cell_links,
                    m_f_backup, m_gf_backup, m_adjc_backup, m_adjv_backup,
                    m_backup_mutex.get(), disjoint_set, cluster_sizes});

        std::optional<traccc::edm::silicon_cluster_collection::buffer>
            cluster_data = std::nullopt;

        // Create the cluster data if requested.
        if (keep_disjoint_set) {

            // Get the number of reconstructed measurements, in an asynchronous way
            // if possible.
            edm::measurement_collection<default_algebra>::buffer::size_type
                num_measurements = 0u;
            if (this->mr().host) {
                const vecmem::async_size size =
                    this->copy().get_size(measurements, *(this->mr().host));
                // Here we could give control back to the caller, once our code
                // allows for it. (coroutines...)<-WIP
                this->await();
                num_measurements = size.get();
            } else {
                num_measurements = this->copy().get_size(measurements);
            }

            // This could be further optimized by only copying the number of
            // elements necessary. But since cluster making is mainly meant for
            // performance measurements, on first order this should be good enough.
            vecmem::vector<unsigned int> cluster_sizes_host =
                ((this->mr().host != nullptr) ? vecmem::vector<unsigned int>(this->mr().host)
                                        : vecmem::vector<unsigned int>());
            this->copy()(cluster_sizes, cluster_sizes_host)->wait();
            cluster_sizes_host.resize(num_measurements);

            // Create the result cluster collection.
            cluster_data.emplace(cluster_sizes_host, this->mr().main, this->mr().host,
                                vecmem::data::buffer_type::resizable);
            this->copy().setup(*cluster_data)->ignore();

            // Run the cluster data reification kernel.
            cluster_maker_kernel(num_cells, disjoint_set, *cluster_data);
        }

        // Return the reconstructed measurements.
        return {std::move(measurements), std::move(cluster_data)};
    }

    /// Clusterization configuration
    config_type m_config;
    /// Memory reserved for edge cases
    mutable vecmem::data::vector_buffer<details::index_t> m_f_backup;
    mutable vecmem::data::vector_buffer<details::index_t> m_gf_backup;
    mutable vecmem::unique_alloc_ptr<unsigned int> m_backup_mutex;
    mutable vecmem::data::vector_buffer<unsigned char> m_adjc_backup;
    mutable vecmem::data::vector_buffer<details::index_t> m_adjv_backup;

};  // class clusterization_algorithm

}  // namespace traccc::device
