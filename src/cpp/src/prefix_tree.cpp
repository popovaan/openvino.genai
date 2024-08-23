// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "prefix_tree.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


namespace ov {
namespace genai {

TrieNode* make_trienode(char data) {
    // Allocate memory for a TrieNode
    TrieNode* node = (TrieNode*) calloc (1, sizeof(TrieNode));
    for (int i=0; i<N; i++)
        node->children[i] = NULL;
    node->is_leaf = 0;
    node->data = data;
    return node;
}

void free_trienode(TrieNode* node) {
    // Free the trienode sequence
    for(int i=0; i<N; i++) {
        if (node->children[i] != NULL) {
            free_trienode(node->children[i]);
        }
        else {
            continue;
        }
    }
    free(node);
}

TrieNode* insert_trie(TrieNode* root, const char* word, size_t length, KVCacheBlock::Ptr block) {
    // Inserts the word onto the Trie
    // ASSUMPTION: The word only has lower case characters
    TrieNode* temp = root;

    for (int i=0; i < length; i++) {
        // Get the relative position in the alphabet list
        int idx = (int) word[i] + 128;
        if (temp->children[idx] == NULL) {
            // If the corresponding child doesn't exist,
            // simply create that child!
            temp->children[idx] = make_trienode(word[i]);
        }
        else {
            // Do nothing. The node already exists
        }
        // Go down a level, to the child referenced by idx
        // since we have a prefix match
        temp = temp->children[idx];
    }
    // At the end of the word, mark this node as the leaf node
    temp->is_leaf = 1;
    temp->block = block;
    return root;
}

void insert_to_prefix_tree(TrieNode* root, const ov::genai::TokenIds& prompt_ids, const ov::genai::TokenIds& generated_ids, KVCacheBlock::Ptr block, size_t content_length) {
    std::vector<int64_t> content;
    content_length = content_length == 0 ? prompt_ids.size() + generated_ids.size() : content_length;
    content.insert( content.end(), prompt_ids.begin(), prompt_ids.begin() + std::min(prompt_ids.size(), content_length));
    if (content_length > prompt_ids.size()) {
        content.insert(content.end(), generated_ids.begin(), generated_ids.begin() + content_length - prompt_ids.size());
    }

    char* data = reinterpret_cast<char*>(content.data());
    std::size_t size = content.size() * sizeof(content[0]);
    auto str = std::string_view(data, size);
    block->set_content(str, size);
    insert_trie(root, data, size, block);
}

void erase_from_prefix_tree(TrieNode* root, const ov::genai::TokenIds& prompt_ids, const ov::genai::TokenIds& generated_ids, size_t content_length) {
    std::vector<int64_t> content;
    content_length = content_length == 0 ? prompt_ids.size() + generated_ids.size() : content_length;
    content.insert( content.end(), prompt_ids.begin(), prompt_ids.begin() + std::min(prompt_ids.size(), content_length));
    if (content_length > prompt_ids.size()) {
        content.insert(content.end(), generated_ids.begin(), generated_ids.begin() + content_length - prompt_ids.size());
    }

    char* data = reinterpret_cast<char*>(content.data());
    std::size_t size = content.size() * sizeof(content[0]);
    delete_trie(root, data, size);
} 


int search_trie(TrieNode* root, char* word)
{
    // Searches for word in the Trie
    TrieNode* temp = root;

    for(int i=0; word[i]!='\0'; i++)
    {
        int position = (int) word[i] + 128;
        if (temp->children[position] == NULL)
            return 0;
        temp = temp->children[position];
    }
    if (temp != NULL && temp->is_leaf == 1)
        return 1;
    return 0;
}


TrieNode* get_trie(TrieNode* root, char* word, size_t size)
{
    // Searches for word in the Trie
    TrieNode* temp = root;

    for(int i=0; i < size; i++)
    {
        int position = (int) word[i] + 128;
        if (temp->children[position] == NULL)
            return NULL;
        temp = temp->children[position];
    }
    if (temp != NULL)
        return temp;
    return NULL;
}

KVCacheBlock::Ptr get_from_prefix_tree(TrieNode* root,  const ov::genai::TokenIds& prompt_ids, const ov::genai::TokenIds& generated_ids, size_t content_length) {
    
    std::vector<int64_t> content;
    content_length = content_length == 0 ? prompt_ids.size() + generated_ids.size() : content_length;
    content.insert( content.end(), prompt_ids.begin(), prompt_ids.begin() + std::min(prompt_ids.size(), content_length));
    if (content_length > prompt_ids.size()) {
        content.insert(content.end(), generated_ids.begin(), generated_ids.begin() + content_length - prompt_ids.size());
    }
    char* data = reinterpret_cast<char*>(content.data());
    std::size_t size = content.size() * sizeof(content[0]);
    auto node = get_trie(root, data, size);
    if (node != NULL) {
        return node->block;
    }
    return nullptr;
}

int check_divergence(TrieNode* root, char* word, size_t size) {
    // Checks if there is branching at the last character of word
    // and returns the largest position in the word where branching occurs
    TrieNode* temp = root;
    int len = size;
    if (len == 0)
        return 0;
    // We will return the largest index where branching occurs
    int last_index = 0;
    for (int i=0; i < len; i++) {
        int position = (int) word[i] + 128;
        if (temp->children[position]) {
            // If a child exists at that position
            // we will check if there exists any other child
            // so that branching occurs
            for (int j=0; j<N; j++) {
                if (j != position && temp->children[j]) {
                    // We've found another child! This is a branch.
                    // Update the branch position
                    last_index = i + 1;
                    break;
                }
            }
            // Go to the next child in the sequence
            temp = temp->children[position];
        }
    }
    return last_index;
}

int find_longest_prefix(TrieNode* root, const char* word, size_t size, char* longest_prefix) {
    // Finds the longest common prefix substring of word
    // in the Trie
    if (!word || size == 0)
        return 0;
    // Length of the longest prefix
    int len = size;

    // We initially set the longest prefix as the word itself,
    // and try to back-tracking from the deepst position to
    // a point of divergence, if it exists
    longest_prefix = (char*) calloc (len, sizeof(char));
    for (int i=0; i < size; i++)
        longest_prefix[i] = word[i];

    // If there is no branching from the root, this
    // means that we're matching the original string!
    // This is not what we want!
    int branch_idx  = check_divergence(root, longest_prefix, size) - 1;
    if (branch_idx >= 0) {
        // There is branching, We must update the position
        // to the longest match and update the longest prefix
        // by the branch index length
        longest_prefix = (char*) realloc (longest_prefix, (branch_idx) * sizeof(char));
    }

    return branch_idx;
}

int is_leaf_node(TrieNode* root, const char* word) {
    // Checks if the prefix match of word and root
    // is a leaf node
    TrieNode* temp = root;
    for (int i=0; word[i]; i++) {
        int position = (int) word[i] + 128;
        if (temp->children[position]) {
            temp = temp->children[position];
        }
    }
    return temp->is_leaf;
}

TrieNode* delete_trie(TrieNode* root, const char* word, size_t size) {
    // Will try to delete the word sequence from the Trie only it 
    // ends up in a leaf node
    if (!root)
        return NULL;
    if (!word ||  size == 0)
        return root;
    // If the node corresponding to the match is not a leaf node,
    // we stop
    if (!is_leaf_node(root, word)) {
        return root;
    }
    TrieNode* temp = root;
    // Find the longest prefix string that is not the current word
    char* longest_prefix;
    auto branch_idx = find_longest_prefix(root, word, size, longest_prefix);
    //printf("Longest Prefix = %s\n", longest_prefix);
    if (branch_idx == 0) {
        free(longest_prefix);
        return root;
    }
    // Keep track of position in the Trie
    int i;
    for (i=0; i < branch_idx; i++) {
        int position = (int) longest_prefix[i];
        if (temp->children[position] != NULL) {
            // Keep moving to the deepest node in the common prefix
            temp = temp->children[position];
        }
        else {
            // There is no such node. Simply return.
            free(longest_prefix);
            return root;
        }
    }
    // Now, we have reached the deepest common node between
    // the two strings. We need to delete the sequence
    // corresponding to word
    int len = size;
    for (; i < len; i++) {
        int position = (int) word[i] + 128;
        if (temp->children[position]) {
            // Delete the remaining sequence
            TrieNode* rm_node = temp->children[position];
            temp->children[position] = NULL;
            free_trienode(rm_node);
        }
    }
    free(longest_prefix);
    return root;
}

void print_trie(TrieNode* root) {
    // Prints the nodes of the trie
    if (!root)
        return;
    TrieNode* temp = root;
    printf("%c -> ", temp->data);
    for (int i=0; i<N; i++) {
        print_trie(temp->children[i]); 
    }
}

void print_search(TrieNode* root, char* word) {
    printf("Searching for %s: ", word);
    if (search_trie(root, word) == 0)
        printf("Not Found\n");
    else
        printf("Found!\n");
}

}
}