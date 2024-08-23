// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <chrono>
#include "sequence_group.hpp"


namespace ov {
namespace genai {
class KVCacheBlock {
    int m_ref_count;
    int m_index;
    size_t m_hash;
    std::chrono::time_point<std::chrono::system_clock> m_timestamp;
    std::string_view m_content;
    size_t m_size;
public:
    ov::genai::TokenIds prompt_ids;
    ov::genai::TokenIds generated_ids;
    using Ptr = std::shared_ptr<KVCacheBlock>;
    using CPtr = std::shared_ptr<const KVCacheBlock>;

    explicit KVCacheBlock(int index)
        : m_ref_count(0),
          m_index(index),
          m_timestamp(std::chrono::system_clock::now()) { }

    int get_index() const {
        return m_index;
    }


    std::string_view get_content() const {
        return  m_content;
    }

    size_t get_size() const {
        return m_size;
    }

    void set_content(std::string_view content, size_t size) {
        m_content = content;
        m_size = size;
    }


    bool is_free() const {
        return m_ref_count == 0;
    }

    void increment() {
        ++m_ref_count;
    }

    void release() {
        if (m_ref_count == 0) {
            std::cout << "k" << std::endl;
        }
        OPENVINO_ASSERT(m_ref_count > 0);
        --m_ref_count;
    }

    bool copy_on_write() const {
        return m_ref_count > 1;
    }

    int get_references_count() const {
        return m_ref_count;
    }

    size_t get_hash() const {
        return m_hash;
    }

    void set_hash(size_t hash) {
        m_hash = hash;
    }

    void set_timestamp(const std::chrono::time_point<std::chrono::system_clock>& timestamp) {
        m_timestamp = timestamp;
    }

    std::chrono::time_point<std::chrono::system_clock> get_timestamp() {
        return m_timestamp;
    }
};


// // The number of children for each node
// // We will construct a N-ary tree and make it
// // a Trie
// // Since we have 26 english letters, we need
// // 26 children per node
const size_t N = 256;

// typedef struct TrieNode TrieNode;

struct TrieNode {
      // The Trie Node Structure
    // Each node has N children, starting from the root
    // and a flag to check if it's a leaf node
    char data; // Storing for printing purposes only
    TrieNode* children[N];
    int is_leaf;
    KVCacheBlock::Ptr block;
    
};

TrieNode* make_trienode(char data);

void free_trienode(TrieNode* node);

TrieNode* insert_trie(TrieNode* root, char* word, KVCacheBlock::Ptr block = nullptr);

void insert_to_prefix_tree(TrieNode* root, const ov::genai::TokenIds& prompt_ids, const ov::genai::TokenIds& generated_ids, KVCacheBlock::Ptr block, size_t content_length=0);

KVCacheBlock::Ptr get_from_prefix_tree(TrieNode* root,  const ov::genai::TokenIds& prompt_ids, const ov::genai::TokenIds& generated_ids, size_t content_length=0);

void erase_from_prefix_tree(TrieNode* root, const ov::genai::TokenIds& prompt_ids, const ov::genai::TokenIds& generated_ids, size_t content_length = 0);

int search_trie(TrieNode* root, char* word);

TrieNode* get_trie(TrieNode* root, char* word);

int find_longest_prefix(TrieNode* root, const char* word, size_t size, char* longest_prefix);

int is_leaf_node(TrieNode* root, const char* word);

TrieNode* delete_trie(TrieNode* root, const char* word, size_t size);

void print_trie(TrieNode* root);

void print_search(TrieNode* root, char* word);
}
}