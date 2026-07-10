// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "llk_sfpu_types.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

// Kernel-invariant SFPU init wrapper. Runs the once-per-kernel SFPU setup (SFPU config register + invariant
// ADDR_MOD_7). Wired into every metal "full init" entry point (compute_kernel_hw_startup, init_sfpu,
// unary_op_init_common, binary_op_init_common). It is idempotent; a kernel may reach more than one entry
// point, so it runs at least once before any SFPU op init on the kernel.
inline void llk_math_sfpu_init_once() { _llk_math_eltwise_unary_sfpu_init_once_(); }

// -------------------------------------------------------------------------------------------------
// Legacy (pre-self-contained) per-op init path.
//
// These two overloads run the generic _llk_math_eltwise_unary_sfpu_init_<Op>() (SFPU config register +
// per-op ADDR_MOD + reset_counters) and then, optionally, the op's init callback. Kept for the ~100 ops that
// have not yet been migrated to the self-contained per-op init. The two forms coexist: an op is either fully
// on the legacy path (these overloads) or fully on the self-contained path (llk_math_eltwise_unary_sfpu_init
// below with a ckernel::sfpu::_init_<op>_ functor).
// -------------------------------------------------------------------------------------------------
template <SfpuType sfpu_op>
inline void llk_math_eltwise_unary_sfpu_init() {
    _llk_math_eltwise_unary_sfpu_init_<sfpu_op>();
}

template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_unary_sfpu_init(F&& init_func, ARGS&&... args) {
    _llk_math_eltwise_unary_sfpu_init_<sfpu_op>();
    init_func(static_cast<ARGS&&>(args)...);
}

// -------------------------------------------------------------------------------------------------
// Self-contained per-op init path (phase 1 pattern).
//
// The op-invariant SFPU config register + ADDR_MOD_7 are hoisted into llk_math_sfpu_init_once() (run once per
// kernel by the metal full-init entry points). A converted op supplies a self-contained functor
// ckernel::sfpu::_init_<op>_ that bakes in its op-dependent ADDR_MOD_6, reset_counters(SET_ABD_F), and the
// body of its old init callback. This overload dispatches straight to that functor with NO generic-init
// prefix -- everything the op needs (minus the hoisted once-init) is inside the functor.
// -------------------------------------------------------------------------------------------------
template <SfpuType sfpu_op, typename F, typename... ARGS>
inline void llk_math_eltwise_unary_sfpu_init_selfcontained(F&& init_func, ARGS&&... args) {
    init_func(std::forward<ARGS>(args)...);
}

}  // namespace ckernel
