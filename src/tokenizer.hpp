#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <cstdint>

// tokenizer.hpp
//
// Implements a character-level tokenizer. Every unique character in the training text
// receives an integer identifier. For the complete works of Shakespeare this vocabulary
// contains sixty-five characters: uppercase letters, lowercase letters, digits,
// punctuation, spaces, and newline characters.
//
// The stoi map converts a character to its integer ID. The itos map reverses the process.
// Both maps are serialised into model.bin so the chat program can decode GPU output
// without access to the original training file.
//
// Production models such as GPT-2 use Byte-Pair Encoding (BPE), which merges frequently
// co-occurring character pairs into single tokens until the vocabulary reaches fifty
// thousand entries. BPE holds roughly four times as much semantic content per context
// window because the word "programming" becomes two tokens rather than eleven. The
// character-level approach here is simpler to implement and sufficient for a single-author
// dataset where vocabulary size is the limiting factor.

struct CharTokenizer {
    int vocab_size = 0;
    std::map<char, int> stoi;  // character to integer ID
    std::map<int, char> itos;  // integer ID to character

    // Scans the training text, collects the set of unique characters, and assigns each
    // one a stable integer ID. std::set iterates in sorted order, so the assignment is
    // deterministic across runs on the same dataset, which is important for reproducibility.
    void build(const std::string& text) {
        std::set<char> chars(text.begin(), text.end());
        stoi.clear(); itos.clear();
        for (char c : chars) {
            int id = (int)itos.size();
            stoi[c] = id;
            itos[id] = c;
        }
        vocab_size = (int)itos.size();
    }

    // Converts a string into a vector of integer token IDs. Characters not present in
    // the vocabulary are silently skipped. This matches the behaviour expected during
    // inference when the user types a character the training set never contained.
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (char c : text) {
            auto it = stoi.find(c);
            if (it != stoi.end()) ids.push_back(it->second);
        }
        return ids;
    }

    // Converts a vector of integer token IDs back into a string. Integer IDs not present
    // in the vocabulary are silently skipped, which handles the case where the model
    // generates an out-of-range index (a sign of numerical instability if it occurs).
    std::string decode(const std::vector<int>& tokens) const {
        std::string result;
        result.reserve(tokens.size());
        for (int id : tokens) {
            auto it = itos.find(id);
            if (it != itos.end()) result += it->second;
        }
        return result;
    }

    // Serialises the vocabulary into an open binary file stream. The format is a 4-byte
    // integer (vocab_size) followed by vocab_size individual bytes, one per character.
    // The characters are written in ID order so that load() can reconstruct both maps by
    // assigning IDs sequentially as it reads.
    void save(std::ofstream& out) const {
        int32_t vs = vocab_size;
        out.write(reinterpret_cast<const char*>(&vs), sizeof(vs));
        for (int i = 0; i < vocab_size; ++i) {
            char c = itos.at(i);
            out.write(&c, 1);
        }
    }

    // Reads the vocabulary from an open binary file stream in the format written by save().
    // Both maps are rebuilt from scratch so that load() can be called on a freshly
    // constructed tokenizer without first calling build().
    void load(std::ifstream& in) {
        int32_t vs;
        in.read(reinterpret_cast<char*>(&vs), sizeof(vs));
        vocab_size = vs;
        stoi.clear(); itos.clear();
        for (int i = 0; i < vocab_size; ++i) {
            char c;
            in.read(&c, 1);
            stoi[c] = i;
            itos[i] = c;
        }
    }
};
