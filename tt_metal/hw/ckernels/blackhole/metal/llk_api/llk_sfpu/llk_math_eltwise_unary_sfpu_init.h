// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "llk_sfpu_types.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

// Kernel-invariant SFPU init: the SFPU config register + invariant ADDR_MOD_7, run once per kernel by the metal
// "full init" entry points (compute_kernel_hw_startup / init_sfpu / unary_op_init_common / binary_op_init_common).
inline void llk_math_sfpu_init_once() { _llk_math_eltwise_unary_sfpu_init_once_(); }

namespace sfpu {
// Self-contained per-op inits for ops used via bare SFPU_UNARY_INIT(OP) (no callback). config_reg + ADDR_MOD_7
// are handled once per kernel by llk_math_sfpu_init_once(); these program only the op's residual state
// (op-specific ADDR_MOD_6 where needed + reset the RWC counters).
inline void abs_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void acos_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void alt_complex_rotate90_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void asin_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void bitwise_and_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void bitwise_not_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void bitwise_or_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void celu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void clamp_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void elu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void fill_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void greater_than_equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void greater_than_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void hardmish_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void hardshrink_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void hardtanh_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void heaviside_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void i0_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void isfinite_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void isinf_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void isnan_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void isneginf_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void isposinf_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void left_shift_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void less_than_equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void less_than_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void logical_not_unary_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void lrelu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void mask_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void negative_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void not_equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void power_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void prelu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void relu_max_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void relu_min_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void reshuffle_rows_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void right_shift_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void selu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void sign_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void softplus_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void softshrink_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void square_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void threshold_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void tiled_prod_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void unary_eq_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void unary_ge_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void unary_gt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void unary_le_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void unary_lt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void unary_ne_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void unused_init() { math::reset_counters(p_setrwc::SET_ABD_F); }
inline void typecast_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

}  // namespace sfpu

// Dependent-false helper so the delegate's else-branch static_assert only fires when an uncategorized bare op is
// instantiated (a plain static_assert(false) in an if-constexpr else is ill-formed pre-C++23).
template <SfpuType>
inline constexpr bool _sfpu_bare_op_unhandled_ = false;

// Bare init entry point: delegates per SFPU op to its self-contained sfpu::<op>_init().
template <SfpuType sfpu_op>
inline void llk_math_eltwise_unary_sfpu_init() {
    if constexpr (sfpu_op == SfpuType::abs) {
        sfpu::abs_init();
    } else if constexpr (sfpu_op == SfpuType::acos) {
        sfpu::acos_init();
    } else if constexpr (sfpu_op == SfpuType::alt_complex_rotate90) {
        sfpu::alt_complex_rotate90_init();
    } else if constexpr (sfpu_op == SfpuType::asin) {
        sfpu::asin_init();
    } else if constexpr (sfpu_op == SfpuType::bitwise_and) {
        sfpu::bitwise_and_init();
    } else if constexpr (sfpu_op == SfpuType::bitwise_not) {
        sfpu::bitwise_not_init();
    } else if constexpr (sfpu_op == SfpuType::bitwise_or) {
        sfpu::bitwise_or_init();
    } else if constexpr (sfpu_op == SfpuType::celu) {
        sfpu::celu_init();
    } else if constexpr (sfpu_op == SfpuType::clamp) {
        sfpu::clamp_init();
    } else if constexpr (sfpu_op == SfpuType::elu) {
        sfpu::elu_init();
    } else if constexpr (sfpu_op == SfpuType::equal_zero) {
        sfpu::equal_zero_init();
    } else if constexpr (sfpu_op == SfpuType::fill) {
        sfpu::fill_init();
    } else if constexpr (sfpu_op == SfpuType::greater_than_equal_zero) {
        sfpu::greater_than_equal_zero_init();
    } else if constexpr (sfpu_op == SfpuType::greater_than_zero) {
        sfpu::greater_than_zero_init();
    } else if constexpr (sfpu_op == SfpuType::hardmish) {
        sfpu::hardmish_init();
    } else if constexpr (sfpu_op == SfpuType::hardshrink) {
        sfpu::hardshrink_init();
    } else if constexpr (sfpu_op == SfpuType::hardtanh) {
        sfpu::hardtanh_init();
    } else if constexpr (sfpu_op == SfpuType::heaviside) {
        sfpu::heaviside_init();
    } else if constexpr (sfpu_op == SfpuType::i0) {
        sfpu::i0_init();
    } else if constexpr (sfpu_op == SfpuType::isfinite) {
        sfpu::isfinite_init();
    } else if constexpr (sfpu_op == SfpuType::isinf) {
        sfpu::isinf_init();
    } else if constexpr (sfpu_op == SfpuType::isnan) {
        sfpu::isnan_init();
    } else if constexpr (sfpu_op == SfpuType::isneginf) {
        sfpu::isneginf_init();
    } else if constexpr (sfpu_op == SfpuType::isposinf) {
        sfpu::isposinf_init();
    } else if constexpr (sfpu_op == SfpuType::left_shift) {
        sfpu::left_shift_init();
    } else if constexpr (sfpu_op == SfpuType::less_than_equal_zero) {
        sfpu::less_than_equal_zero_init();
    } else if constexpr (sfpu_op == SfpuType::less_than_zero) {
        sfpu::less_than_zero_init();
    } else if constexpr (sfpu_op == SfpuType::logical_not_unary) {
        sfpu::logical_not_unary_init();
    } else if constexpr (sfpu_op == SfpuType::lrelu) {
        sfpu::lrelu_init();
    } else if constexpr (sfpu_op == SfpuType::mask) {
        sfpu::mask_init();
    } else if constexpr (sfpu_op == SfpuType::negative) {
        sfpu::negative_init();
    } else if constexpr (sfpu_op == SfpuType::not_equal_zero) {
        sfpu::not_equal_zero_init();
    } else if constexpr (sfpu_op == SfpuType::power) {
        sfpu::power_init();
    } else if constexpr (sfpu_op == SfpuType::prelu) {
        sfpu::prelu_init();
    } else if constexpr (sfpu_op == SfpuType::relu_max) {
        sfpu::relu_max_init();
    } else if constexpr (sfpu_op == SfpuType::relu_min) {
        sfpu::relu_min_init();
    } else if constexpr (sfpu_op == SfpuType::reshuffle_rows) {
        sfpu::reshuffle_rows_init();
    } else if constexpr (sfpu_op == SfpuType::right_shift) {
        sfpu::right_shift_init();
    } else if constexpr (sfpu_op == SfpuType::selu) {
        sfpu::selu_init();
    } else if constexpr (sfpu_op == SfpuType::sign) {
        sfpu::sign_init();
    } else if constexpr (sfpu_op == SfpuType::softplus) {
        sfpu::softplus_init();
    } else if constexpr (sfpu_op == SfpuType::softshrink) {
        sfpu::softshrink_init();
    } else if constexpr (sfpu_op == SfpuType::square) {
        sfpu::square_init();
    } else if constexpr (sfpu_op == SfpuType::threshold) {
        sfpu::threshold_init();
    } else if constexpr (sfpu_op == SfpuType::tiled_prod) {
        sfpu::tiled_prod_init();
    } else if constexpr (sfpu_op == SfpuType::unary_eq) {
        sfpu::unary_eq_init();
    } else if constexpr (sfpu_op == SfpuType::unary_ge) {
        sfpu::unary_ge_init();
    } else if constexpr (sfpu_op == SfpuType::unary_gt) {
        sfpu::unary_gt_init();
    } else if constexpr (sfpu_op == SfpuType::unary_le) {
        sfpu::unary_le_init();
    } else if constexpr (sfpu_op == SfpuType::unary_lt) {
        sfpu::unary_lt_init();
    } else if constexpr (sfpu_op == SfpuType::unary_ne) {
        sfpu::unary_ne_init();
    } else if constexpr (sfpu_op == SfpuType::unused) {
        sfpu::unused_init();
    } else if constexpr (sfpu_op == SfpuType::typecast) {
        sfpu::typecast_init();
    } else {
        static_assert(
            _sfpu_bare_op_unhandled_<sfpu_op>,
            "Bare SFPU_UNARY_INIT(OP) used for an op with no sfpu::<op>_init(). Add its self-contained init and a "
            "delegate branch in llk_math_eltwise_unary_sfpu_init.h.");
    }
}

// Callback init entry point (SFPU_UNARY_INIT_FN / _FN_ARGS / two-arg SFPU_UNARY_INIT): the op-specific init_func
// is itself self-contained (it programs its own ADDR_MOD_6 if needed + resets counters). No generic prefix.
template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_unary_sfpu_init(F&& init_func, ARGS&&... args) {
    init_func(std::forward<ARGS>(args)...);
}

}  // namespace ckernel
