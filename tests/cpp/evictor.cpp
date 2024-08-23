// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "scheduler.hpp"
#include <chrono>
#include <thread>

TEST(TestEvictor, general_test) {
    ov::genai::Evictor evictor;
    auto block0 = std::make_shared<ov::genai::KVCacheBlock>(0);
    block0->set_hash(77);
    block0->prompt_ids = {0,1,3};
    block0->generated_ids = {5,6,7};
    std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(1));
    auto block1 = std::make_shared<ov::genai::KVCacheBlock>(1);
    block1->set_hash(56);
    block1->prompt_ids = {5,2};
    block1->generated_ids = {5};
    std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(1));
    auto block2 = std::make_shared<ov::genai::KVCacheBlock>(2);
    block2->set_hash(23);
    block2->prompt_ids = {7,8,9};
    block2->generated_ids = {7};
    std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(1));
    evictor.add(block0);
    evictor.add(block1);
    evictor.add(block2);
    EXPECT_EQ(evictor.num_blocks(), 3);

    auto block = evictor.get_block({5,2, 5}, 3);
    EXPECT_EQ(block->get_index(), 1);
    EXPECT_EQ(block->get_hash(), 56);
    EXPECT_EQ(block->get_references_count(), 1);
    EXPECT_EQ(evictor.num_blocks(), 2);

    EXPECT_EQ(evictor.get_block({1,1,1}, 3), nullptr);
    EXPECT_EQ(evictor.num_blocks(), 2);

    EXPECT_EQ(evictor.get_lru_block()->get_index(), 0);
    EXPECT_EQ(evictor.num_blocks(), 1);

    auto block3 = std::make_shared<ov::genai::KVCacheBlock>(7);
    block3->prompt_ids = {2,2};
    block3->generated_ids = {};
    std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(1));
    auto block4 = std::make_shared<ov::genai::KVCacheBlock>(10);
    block4->set_hash(99);
    block4->prompt_ids = {6,8,10};
    block4->generated_ids = {5,1};
    std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(1));
    evictor.add(block3);
    evictor.add(block4);

    EXPECT_EQ(evictor.get_lru_block()->get_index(), 2);
    EXPECT_EQ(evictor.get_lru_block()->get_index(), 7);
    EXPECT_EQ(evictor.get_lru_block()->get_index(), 10);
    EXPECT_EQ(evictor.get_lru_block(), nullptr);
    EXPECT_EQ(evictor.num_blocks(), 0);
}
