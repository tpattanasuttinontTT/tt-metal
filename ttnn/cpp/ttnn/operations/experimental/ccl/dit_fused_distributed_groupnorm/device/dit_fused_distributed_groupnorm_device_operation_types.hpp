// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt_stl/reflection.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

// Mirrors GroupNormParams, plus CCL fields for stats all-gather on cluster_axis.
struct DitFusedDistributedGroupnormParams {
    float eps = 0.0f;
    uint32_t num_groups = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    bool use_welford = false;

    uint32_t cluster_axis = 0;
    uint32_t num_links = 1;
    uint32_t ring_size = 1;
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear;
    std::vector<GlobalSemaphore> multi_device_global_semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("eps", eps);
        attrs.emplace_back("num_groups", num_groups);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        attrs.emplace_back("use_welford", use_welford);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("topology", topology);
        return attrs;
    }
};

// Mirrors GroupNormInputs (gamma/beta), plus persistent stats buffer for AG.
struct DitFusedDistributedGroupnormInputs {
    Tensor input;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
    std::optional<Tensor> input_mask;
    std::optional<Tensor> persistent_output_buffer;
};

// Stats stick: per group bf16 [mean, var]. stick_bytes = round_up(num_groups * 4, 64) — a multiple
// of NOC_DRAM_READ_ALIGNMENT_BYTES (64 on Blackhole) because the stick is the per-device page of the
// gathered stats DRAM buffer, and non-64-aligned page offsets read back as zero. total_pages = ring_size.
struct DitFusedDistributedGroupnormSizing {
    bool is_local = false;
    uint32_t num_groups = 0;
    uint32_t stick_bytes = 0;
    uint32_t total_pages = 0;
    uint32_t page_size_bytes = 0;
};

DitFusedDistributedGroupnormSizing compute_sizing(const DitFusedDistributedGroupnormParams& args, const Tensor& input);

}  // namespace ttnn::experimental::prim
