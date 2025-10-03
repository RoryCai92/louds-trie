#include "louds-trie.hpp"

#include <cassert>
#include <vector>

// Platform detection and intrinsic includes
#if defined(_MSC_VER)
 #include <intrin.h>
 #if defined(_M_X64) || defined(_M_IX86)
  #include <immintrin.h>
  #define HAS_PDEP 1
 #endif
#elif defined(__GNUC__) || defined(__clang__)
 #if defined(__x86_64__) || defined(__i386__)
  #include <x86intrin.h>
  #define HAS_PDEP 1
 #endif
#endif

#ifndef HAS_PDEP
 #define HAS_PDEP 0
#endif

namespace louds {
namespace {

uint64_t Popcnt(uint64_t x) {
#ifdef _MSC_VER
  return __popcnt64(x);
#else  // _MSC_VER
  return __builtin_popcountll(x);
#endif  // _MSC_VER
}

uint64_t Ctz(uint64_t x) {
#ifdef _MSC_VER
  return _tzcnt_u64(x);
#else  // _MSC_VER
  return __builtin_ctzll(x);
#endif  // _MSC_VER
}

// Portable implementation of _pdep_u64 for non-x86 architectures
uint64_t Pdep(uint64_t src, uint64_t mask) {
#if HAS_PDEP
  return _pdep_u64(src, mask);
#else
  // Portable implementation using bit manipulation
  uint64_t result = 0;
  while (mask != 0) {
    uint64_t mask_bit = mask & -mask;  // Extract lowest set bit
    if (src & 1) {
      result |= mask_bit;
    }
    src >>= 1;
    mask ^= mask_bit;  // Clear the bit we just processed
  }
  return result;
#endif
}

struct BitVector {
  struct Rank {
    uint32_t abs_hi;
    uint8_t abs_lo;
    uint8_t rels[3];

    uint64_t abs() const {
      return ((uint64_t)abs_hi << 8) | abs_lo;
    }
    void set_abs(uint64_t abs) {
      abs_hi = (uint32_t)(abs >> 8);
      abs_lo = (uint8_t)abs;
    }
  };

  vector<uint64_t> words;
  vector<Rank> ranks;
  vector<uint32_t> selects;
  uint64_t n_bits;

  BitVector() : words(), ranks(), selects(), n_bits(0) {}

  uint64_t get(uint64_t i) const {
    return (words[i / 64] >> (i % 64)) & 1UL;
  }
  void set(uint64_t i, uint64_t bit) {
    if (bit) {
      words[i / 64] |= (1UL << (i % 64));
    } else {
      words[i / 64] &= ~(1UL << (i % 64));
    }
  }

  void add(uint64_t bit) {
    if (n_bits % 256 == 0) {
      words.resize((n_bits + 256) / 64);
    }
    set(n_bits, bit);
    ++n_bits;
  }
  // build builds indexes for rank and select.
  void build() {
    uint64_t n_blocks = words.size() / 4;
    uint64_t n_ones = 0;
    ranks.resize(n_blocks + 1);
    for (uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
      ranks[block_id].set_abs(n_ones);
      for (uint64_t j = 0; j < 4; ++j) {
        if (j != 0) {
          uint64_t rel = n_ones - ranks[block_id].abs();
          ranks[block_id].rels[j - 1] = rel;
        }

        uint64_t word_id = (block_id * 4) + j;
        uint64_t word = words[word_id];
        uint64_t n_pops = Popcnt(word);
        uint64_t new_n_ones = n_ones + n_pops;
        if (((n_ones + 255) / 256) != ((new_n_ones + 255) / 256)) {
          uint64_t count = n_ones;
          while (word != 0) {
            uint64_t pos = Ctz(word);
            if (count % 256 == 0) {
              selects.push_back(((word_id * 64) + pos) / 256);
              break;
            }
            word ^= 1UL << pos;
            ++count;
          }
        }
        n_ones = new_n_ones;
      }
    }
    ranks.back().set_abs(n_ones);
    selects.push_back(words.size() * 64 / 256);
  }

  // rank returns the number of 1-bits in the range [0, i).
  uint64_t rank(uint64_t i) const {
    uint64_t word_id = i / 64;
    uint64_t bit_id = i % 64;
    uint64_t rank_id = word_id / 4;
    uint64_t rel_id = word_id % 4;
    uint64_t n = ranks[rank_id].abs();
    if (rel_id != 0) {
      n += ranks[rank_id].rels[rel_id - 1];
    }
    n += Popcnt(words[word_id] & ((1UL << bit_id) - 1));
    return n;
  }
  // select returns the position of the (i+1)-th 1-bit.
  uint64_t select(uint64_t i) const {
    const uint64_t block_id = i / 256;
    uint64_t begin = selects[block_id];
    uint64_t end = selects[block_id + 1] + 1UL;
    if (begin + 10 >= end) {
      while (i >= ranks[begin + 1].abs()) {
        ++begin;
      }
    } else {
      while (begin + 1 < end) {
        const uint64_t middle = (begin + end) / 2;
        if (i < ranks[middle].abs()) {
          end = middle;
        } else {
          begin = middle;
        }
      }
    }
    const uint64_t rank_id = begin;
    i -= ranks[rank_id].abs();

    uint64_t word_id = rank_id * 4;
    if (i < ranks[rank_id].rels[1]) {
      if (i >= ranks[rank_id].rels[0]) {
        word_id += 1;
        i -= ranks[rank_id].rels[0];
      }
    } else if (i < ranks[rank_id].rels[2]) {
      word_id += 2;
      i -= ranks[rank_id].rels[1];
    } else {
      word_id += 3;
      i -= ranks[rank_id].rels[2];
    }
    return (word_id * 64) + Ctz(Pdep(1UL << i, words[word_id]));
  }

  uint64_t size() const {
    return sizeof(uint64_t) * words.size()
      + sizeof(Rank) * ranks.size()
      + sizeof(uint32_t) * selects.size();
  }
};

struct Level {
  BitVector louds;
  BitVector outs;
  vector<uint8_t> labels;
  uint64_t offset;

  Level() : louds(), outs(), labels(), offset(0) {}

  uint64_t size() const;
};

uint64_t Level::size() const {
  return louds.size() + outs.size() + labels.size();
}

}  // namespace

class TrieImpl {
 public:
  TrieImpl();
  ~TrieImpl() {}

  void add(const string &key);
  void build();
  int64_t lookup(const string &query) const;
  
  static TrieImpl* merge(const TrieImpl& trie1, const TrieImpl& trie2);

  uint64_t n_keys() const {
    return n_keys_;
  }
  uint64_t n_nodes() const {
    return n_nodes_;
  }
  uint64_t size() const {
    return size_;
  }
  
  // struct for child range
  struct ChildRange {
    uint64_t begin;  
    uint64_t end;    
  };
  
  ChildRange get_child_range(uint64_t depth, uint64_t node_id) const;

 private:
  vector<Level> levels_;
  uint64_t n_keys_;
  uint64_t n_nodes_;
  uint64_t size_;
  string last_key_;
  
  friend class Trie;
};

TrieImpl::TrieImpl()
  : levels_(2), n_keys_(0), n_nodes_(1), size_(0), last_key_() {
  levels_[0].louds.add(0);
  levels_[0].louds.add(1);
  levels_[1].louds.add(1);
  levels_[0].outs.add(0);
  levels_[0].labels.push_back(' ');
}

void TrieImpl::add(const string &key) {
  assert(key > last_key_);
  if (key.empty()) {
    levels_[0].outs.set(0, 1);
    ++levels_[1].offset;
    ++n_keys_;
    return;
  }
  if (key.length() + 1 >= levels_.size()) {
    levels_.resize(key.length() + 2);
  }
  uint64_t i = 0;
  for ( ; i < key.length(); ++i) {
    Level &level = levels_[i + 1];
    uint8_t byte = key[i];
    if ((i == last_key_.length()) || (byte != level.labels.back())) {
      level.louds.set(levels_[i + 1].louds.n_bits - 1, 0);
      level.louds.add(1);
      level.outs.add(0);
      level.labels.push_back(key[i]);
      ++n_nodes_;
      break;
    }
  }
  for (++i; i < key.length(); ++i) {
    Level &level = levels_[i + 1];
    level.louds.add(0);
    level.louds.add(1);
    level.outs.add(0);
    level.labels.push_back(key[i]);
    ++n_nodes_;
  }
  levels_[key.length() + 1].louds.add(1);
  ++levels_[key.length() + 1].offset;
  levels_[key.length()].outs.set(levels_[key.length()].outs.n_bits - 1, 1);
  ++n_keys_;
  last_key_ = key;
}

void TrieImpl::build() {
  uint64_t offset = 0;
  for (uint64_t i = 0; i < levels_.size(); ++i) {
    Level &level = levels_[i];
    level.louds.build();
    level.outs.build();
    offset += levels_[i].offset;
    level.offset = offset;
    size_ += level.size();
  }
}

int64_t TrieImpl::lookup(const string &query) const {
  if (query.length() >= levels_.size()) {
    return false;
  }
  uint64_t node_id = 0;
  for (uint64_t i = 0; i < query.length(); ++i) {
    const Level &level = levels_[i + 1];
    uint64_t node_pos;
    if (node_id != 0) {
      node_pos = level.louds.select(node_id - 1) + 1;
      node_id = node_pos - node_id;
    } else {
      node_pos = 0;
    }

    // Linear search implementation
    // for (uint8_t byte = query[i]; ; ++node_pos, ++node_id) {
    //   if (level.louds.get(node_pos) || level.labels[node_id] > byte) {
    //     return -1;
    //   }
    //   if (level.labels[node_id] == byte) {
    //     break;
    //   }
    // }

    // Binary search implementation
    uint64_t end = node_pos;
    uint64_t word = level.louds.words[end / 64] >> (end % 64);
    if (word == 0) {
      end += 64 - (end % 64);
      word = level.louds.words[end / 64];
      while (word == 0) {
        end += 64;
        word = level.louds.words[end / 64];
      }
    }
    end += Ctz(word);
    uint64_t begin = node_id;
    end = begin + end - node_pos;

    uint8_t byte = query[i];
    while (begin < end) {
      node_id = (begin + end) / 2;
      if (byte < level.labels[node_id]) {
        end = node_id;
      } else if (byte > level.labels[node_id]) {
        begin = node_id + 1;
      } else {
        break;
      }
    }
    if (begin >= end) {
      return -1;
    }
  }
  const Level &level = levels_[query.length()];
  if (!level.outs.get(node_id)) {
    return false;
  }
  return level.offset + level.outs.rank(node_id);
}

Trie::Trie() : impl_(new TrieImpl) {}

Trie::~Trie() {
  delete impl_;
}

void Trie::add(const string &key) {
  return impl_->add(key);
}

void Trie::build() {
  impl_->build();
}

int64_t Trie::lookup(const string &query) const {
  return impl_->lookup(query);
}

uint64_t Trie::n_keys() const {
  return impl_->n_keys();
}

uint64_t Trie::n_nodes() const {
  return impl_->n_nodes();
}

uint64_t Trie::size() const {
  return impl_->size();
}

// get_child_range helper
TrieImpl::ChildRange TrieImpl::get_child_range(uint64_t depth, uint64_t node_id) const {
  if (depth >= levels_.size()) {
    return {0, 0};
  }
  
  const Level& level = levels_[depth];
  
  // find start
  uint64_t run_start;
  if (node_id != 0) {
    run_start = level.louds.select(node_id - 1) + 1;
  } else {
    run_start = 0;
  }
  
  // find end 
  uint64_t run_end = run_start;
  uint64_t word = level.louds.words[run_end / 64] >> (run_end % 64);
  if (word == 0) {
    run_end += 64 - (run_end % 64);
    word = level.louds.words[run_end / 64];
    while (word == 0) {
      run_end += 64;
      word = level.louds.words[run_end / 64];
    }
  }
  run_end += Ctz(word);
  
  uint64_t begin = run_start - node_id;  
  uint64_t end = begin + (run_end - run_start);  
  
  return {begin, end};
}

// merge function 
TrieImpl* TrieImpl::merge(const TrieImpl& trie1, const TrieImpl& trie2) {
  TrieImpl* merged = new TrieImpl();
  
  merged->levels_.clear();
  merged->n_nodes_ = 0;
  merged->n_keys_ = 0;
  
  size_t max_depth = std::max(trie1.levels_.size(), trie2.levels_.size());
  merged->levels_.resize(max_depth);
  
  // struct to store both node
  struct NodePair {
    uint64_t node1;
    uint64_t node2;
    bool has1;
    bool has2;
  };
  
  vector<vector<NodePair>> queue(max_depth);
  queue[0].push_back({0, 0, true, true});
  
  // root level
  merged->levels_[0].louds.add(0);
  merged->levels_[0].louds.add(1);
  merged->levels_[0].labels.push_back(' ');
  merged->n_nodes_ = 1;
  
  // go through each level
  for (size_t depth = 0; depth < max_depth - 1; ++depth) {
    if (queue[depth].empty()) continue;
    
    Level& next_level = merged->levels_[depth + 1];
    
    // iterate all the parent node
    for (const NodePair& parent_pair : queue[depth]) {
      if (!parent_pair.has1 && !parent_pair.has2) continue;
      
      ChildRange range1 = {0, 0};
      ChildRange range2 = {0, 0};
      
      if (parent_pair.has1 && depth + 1 < trie1.levels_.size()) {
        range1 = trie1.get_child_range(depth + 1, parent_pair.node1);
      }
      if (parent_pair.has2 && depth + 1 < trie2.levels_.size()) {
        range2 = trie2.get_child_range(depth + 1, parent_pair.node2);
      }
      
      uint64_t i1 = range1.begin;
      uint64_t i2 = range2.begin;
      
      while (i1 < range1.end || i2 < range2.end) {
        uint8_t label1;
        if (i1 < range1.end && depth + 1 < trie1.levels_.size()) {
          label1 = trie1.levels_[depth + 1].labels[i1];
        } else {
          label1 = 255;
        }
        
        uint8_t label2;
        if (i2 < range2.end && depth + 1 < trie2.levels_.size()) {
          label2 = trie2.levels_[depth + 1].labels[i2];
        } else {
          label2 = 255;
        }
        
        bool has1 = (i1 < range1.end);
        bool has2 = (i2 < range2.end);
        
        if (has1 && has2 && label1 == label2) {
          // same label in both
          next_level.louds.add(0); 
          next_level.labels.push_back(label1);
          
          // take the or of terminal bit
          bool term1 = trie1.levels_[depth + 1].outs.get(i1);
          bool term2 = trie2.levels_[depth + 1].outs.get(i2);
          if (term1 || term2) {
            next_level.outs.add(1);
          } else {
            next_level.outs.add(0);
          }
          
          if (term1 || term2) {
            merged->n_keys_++;
            next_level.offset++;
          }
          
          // add to the queue for next level
          if (depth + 2 < max_depth) {
            queue[depth + 1].push_back({i1, i2, true, true});
          }
          
          i1++;
          i2++;
        } else if (!has2 || (has1 && label1 < label2)) {
          // take trie1
          next_level.louds.add(0);
          next_level.labels.push_back(label1);
          
          bool term1 = trie1.levels_[depth + 1].outs.get(i1);
          if (term1) {
            next_level.outs.add(1);
          } else {
            next_level.outs.add(0);
          }
          
          if (term1) {
            merged->n_keys_++;
            next_level.offset++;
          }
          
          if (depth + 2 < max_depth) {
            queue[depth + 1].push_back({i1, 0, true, false});
          }
          
          i1++;
        } else {
          // take trie2
          next_level.louds.add(0);
          next_level.labels.push_back(label2);
          
          bool term2 = trie2.levels_[depth + 1].outs.get(i2);
          if (term2) {
            next_level.outs.add(1);
          } else {
            next_level.outs.add(0);
          }
          
          if (term2) {
            merged->n_keys_++;
            next_level.offset++;
          }
          
          // Enqueue child from trie2 only
          if (depth + 2 < max_depth) {
            queue[depth + 1].push_back({0, i2, false, true});
          }
          
          i2++;
        }
        
        merged->n_nodes_++;
      }
      
      // add run terminator
      next_level.louds.add(1);
    }
  }
  
  merged->build();
  
  return merged;
}

Trie* Trie::merge_trie(const Trie& trie1, const Trie& trie2) {
  Trie* merged = new Trie();
  delete merged->impl_;
  merged->impl_ = TrieImpl::merge(*trie1.impl_, *trie2.impl_);
  return merged;
}

}  // namespace louds
