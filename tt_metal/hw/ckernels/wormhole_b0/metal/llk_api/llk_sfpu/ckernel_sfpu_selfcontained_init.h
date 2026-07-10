// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// Self-contained SFPU per-op init functors (phase 1 pattern).
//
// Each converted op exposes a single ckernel::sfpu::_init_<op>_() that is self-contained: it programs the
// op-dependent ADDR_MOD_6 (a no-op for ops that don't need it), runs math::reset_counters(SET_ABD_F), and then
// the body of the op's old init callback. The op-invariant SFPU config register + ADDR_MOD_7 are NOT here --
// they are hoisted into llk_math_sfpu_init_once() and run once per kernel by the metal full-init entry points
// (compute_kernel_hw_startup / init_sfpu / unary_op_init_common / binary_op_init_common). The SFPU_UNARY_INIT_SC*
// macros dispatch to these functors via llk_math_eltwise_unary_sfpu_init_selfcontained (no generic-init prefix).
//
// This header holds the phase-1 converted ops (abs, exponential, hardsigmoid). The ~100-op fan-out adds more
// _init_<op>_ functors here following the same pattern.
//

#include "ckernel_sfpu_activations.h"     // sfpu::hardsigmoid_init
#include "ckernel_sfpu_exp.h"             // sfpu::exp_init
#include "llk_math_eltwise_unary_sfpu.h"  // _llk_math_eltwise_unary_sfpu_configure_op_addrmod_, math::reset_counters

namespace ckernel {
namespace sfpu {

// Common tail of every self-contained per-op init: op-dependent ADDR_MOD_6 + reset the RWC counters. The op's
// own init body (if any) runs after this in the per-op _init_<op>_ below.
template <SfpuType sfpu_op>
inline void _sfpu_op_addrmod_reset_() {
    _llk_math_eltwise_unary_sfpu_configure_op_addrmod_<sfpu_op>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// abs: bare op (no old init callback). abs is not in any ADDR_MOD_6 list, so this is just addrmod (no-op)
// + reset_counters.
inline void _init_abs_() { _sfpu_op_addrmod_reset_<SfpuType::abs>(); }

// exponential: templated-callback op. Mirrors the args the old SFPU_UNARY_INIT_FN callback took.
template <
    bool APPROXIMATION_MODE,
    uint32_t scale = 0x3F800000,
    bool CLAMP_NEGATIVE = true,
    bool is_fp32_dest_acc_en = false>
inline void _init_exp_() {
    _sfpu_op_addrmod_reset_<SfpuType::exponential>();
    exp_init<APPROXIMATION_MODE, scale, CLAMP_NEGATIVE, is_fp32_dest_acc_en>();
}

// hardsigmoid: templated-callback op.
template <bool APPROXIMATION_MODE>
inline void _init_hardsigmoid_() {
    _sfpu_op_addrmod_reset_<SfpuType::hardsigmoid>();
    hardsigmoid_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
