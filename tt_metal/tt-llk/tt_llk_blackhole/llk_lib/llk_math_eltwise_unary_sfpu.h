// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_sfpu_types.h"

using namespace ckernel;

// local function declarations
template <SfpuType sfpu_op>
inline void eltwise_unary_sfpu_configure_addrmod()
{
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    if constexpr (sfpu_op == SfpuType::topk_local_sort)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 32},
        }
            .set(ADDR_MOD_6);
    }

    if constexpr (
        sfpu_op == SfpuType::reciprocal || sfpu_op == SfpuType::typecast || sfpu_op == SfpuType::unary_max || sfpu_op == SfpuType::unary_min ||
        sfpu_op == SfpuType::unary_max_int32 || sfpu_op == SfpuType::unary_min_int32 || sfpu_op == SfpuType::unary_max_uint32 ||
        sfpu_op == SfpuType::unary_min_uint32 || sfpu_op == SfpuType::signbit || sfpu_op == SfpuType::equal_zero || sfpu_op == SfpuType::not_equal_zero ||
        sfpu_op == SfpuType::less_than_zero || sfpu_op == SfpuType::greater_than_equal_zero || sfpu_op == SfpuType::less_than_equal_zero ||
        sfpu_op == SfpuType::greater_than_zero)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_6);
    }
}

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    eltwise_unary_sfpu_configure_addrmod<sfpu_op>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// Kernel-invariant SFPU init. Runs the parts of the per-op init that are identical for every SFPU op and
// therefore only need to run once per kernel: the SFPU config register (SFPCONFIG(0, 0xF, 1)) and the
// invariant ADDR_MOD_7 = {srca:0, srcb:0, dest:0}. Hoisted out of the per-op path so that the self-contained
// per-op init (ckernel::sfpu::_init_<op>_) does not re-run these on every op init. Metal wires this into every
// "full init" entry point (compute_kernel_hw_startup, init_sfpu, unary_op_init_common, binary_op_init_common),
// and the tt-llk standalone SFPU test harness wires it into its init prelude.
inline void _llk_math_eltwise_unary_sfpu_init_once_()
{
    sfpu::_init_sfpu_config_reg();

    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
}

// Per-op ADDR_MOD_6 programming, split out of eltwise_unary_sfpu_configure_addrmod() (which also sets the now
// hoisted ADDR_MOD_7). Exhaustively categorizes every op that uses the self-contained init path
// (ckernel::sfpu::_init_<op>_): ops needing a non-default ADDR_MOD_6 set it here, ops that need none are listed
// explicitly, and any uncategorized op is a compile error -- so converting a new op forces a conscious
// ADDR_MOD_6 decision instead of silently defaulting.
template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_configure_op_addrmod_()
{
    constexpr bool topk = sfpu_op == SfpuType::topk_local_sort;

    constexpr bool dest_incr_2 = sfpu_op == SfpuType::reciprocal || sfpu_op == SfpuType::typecast || sfpu_op == SfpuType::unary_max ||
                                 sfpu_op == SfpuType::unary_min || sfpu_op == SfpuType::unary_max_int32 || sfpu_op == SfpuType::unary_min_int32 ||
                                 sfpu_op == SfpuType::unary_max_uint32 || sfpu_op == SfpuType::unary_min_uint32 || sfpu_op == SfpuType::signbit ||
                                 sfpu_op == SfpuType::equal_zero || sfpu_op == SfpuType::not_equal_zero || sfpu_op == SfpuType::less_than_zero ||
                                 sfpu_op == SfpuType::greater_than_equal_zero || sfpu_op == SfpuType::less_than_equal_zero ||
                                 sfpu_op == SfpuType::greater_than_zero;

    constexpr bool no_addr_mod_6 = sfpu_op == SfpuType::abs || sfpu_op == SfpuType::exponential || sfpu_op == SfpuType::hardsigmoid;

    static_assert(
        topk || dest_incr_2 || no_addr_mod_6,
        "SFPU op is not categorized for the self-contained init ADDR_MOD_6 path. Add it to the topk / "
        "dest-incr-2 / no-ADDR_MOD_6 list in _llk_math_eltwise_unary_sfpu_configure_op_addrmod_.");

    if constexpr (topk)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 32},
        }
            .set(ADDR_MOD_6);
    }
    else if constexpr (dest_incr_2)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_6);
    }
}
