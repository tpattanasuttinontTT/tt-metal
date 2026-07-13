// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <tt-metalium/float8.hpp>

TEST(Float8E4M3, EncodesExponent15FiniteRange) {
    EXPECT_EQ(float8_e4m3(240.0f).to_bits(), 0x77);
    EXPECT_EQ(float8_e4m3(248.0f).to_bits(), 0x78);  // RNE carry into exponent 15
    EXPECT_EQ(float8_e4m3(256.0f).to_bits(), 0x78);
    EXPECT_EQ(float8_e4m3(384.0f).to_bits(), 0x7C);
    EXPECT_EQ(float8_e4m3(448.0f).to_bits(), 0x7E);
    EXPECT_EQ(float8_e4m3(-448.0f).to_bits(), 0xFE);
}

TEST(Float8E4M3, SaturatesOverflowAndPreservesNan) {
    EXPECT_EQ(float8_e4m3(449.0f).to_bits(), 0x7E);
    EXPECT_EQ(float8_e4m3(std::numeric_limits<float>::infinity()).to_bits(), 0x7E);
    EXPECT_EQ(float8_e4m3(-std::numeric_limits<float>::infinity()).to_bits(), 0xFE);
    EXPECT_EQ(float8_e4m3(std::numeric_limits<float>::quiet_NaN()).to_bits(), 0x7F);
}

TEST(Float8E4M3, DecodesExponent15FiniteRange) {
    EXPECT_EQ(static_cast<float>(float8_e4m3::from_bits(0x77)), 240.0f);
    EXPECT_EQ(static_cast<float>(float8_e4m3::from_bits(0x78)), 256.0f);
    EXPECT_EQ(static_cast<float>(float8_e4m3::from_bits(0x7C)), 384.0f);
    EXPECT_EQ(static_cast<float>(float8_e4m3::from_bits(0x7E)), 448.0f);
    EXPECT_EQ(static_cast<float>(float8_e4m3::from_bits(0xFE)), -448.0f);
    EXPECT_TRUE(std::isnan(static_cast<float>(float8_e4m3::from_bits(0x7F))));
    EXPECT_TRUE(std::isnan(static_cast<float>(float8_e4m3::from_bits(0xFF))));
}
