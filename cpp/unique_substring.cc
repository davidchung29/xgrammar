/*!
 *  Copyright (c) 2026 by Contributors
 * \file unique_substring.cc
 * \brief Runtime-bound unique-substring matcher implementation.
 */

#include <xgrammar/matcher.h>
#include <xgrammar/unique_substring.h>

#include <cstdint>
#include <cstring>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "suffix_automata.h"
#include "support/dynamic_bitset.h"
#include "support/encoding.h"
#include "support/logging.h"

namespace xgrammar {

// Shared with GrammarMatcher so both matchers accept the same DLPack bitmask layout.
int32_t* CheckAndGetBitmaskPtr(const DLTensor& token_bitmask, int vocab_size, int index);

class UniqueSubstringMatcher::Impl {
 public:
  Impl(const std::string& source, const TokenizerInfo& tokenizer_info)
      : tokenizer_info_(tokenizer_info), special_token_ids_(tokenizer_info.GetSpecialTokenIds()) {
    BuildIndex(source);
  }

  bool AcceptToken(int32_t token_id) {
    if (terminated_ || token_id < 0 || token_id >= tokenizer_info_.GetVocabSize()) {
      return false;
    }
    if (IsStopToken(token_id)) {
      if (!IsCompleted()) return false;
      history_.push_back({cursor_, terminated_});
      terminated_ = true;
      return true;
    }
    if (IsSpecialToken(token_id)) return false;
    const auto& vocab = tokenizer_info_.GetDecodedVocab();
    if (token_id >= static_cast<int32_t>(vocab.size()) || vocab[token_id].empty()) return false;
    auto next_cursor = Walk(cursor_, vocab[token_id]);
    if (!next_cursor.has_value()) return false;
    history_.push_back({cursor_, terminated_});
    cursor_ = *next_cursor;
    return true;
  }

  bool AcceptString(const std::string& input) {
    if (terminated_ || input.empty()) return false;
    auto next_cursor = Walk(cursor_, input);
    if (!next_cursor.has_value()) return false;
    history_.push_back({cursor_, terminated_});
    cursor_ = *next_cursor;
    return true;
  }

  bool FillNextTokenBitmask(DLTensor* next_token_bitmask, int index) {
    XGRAMMAR_CHECK(!terminated_
    ) << "UniqueSubstringMatcher has terminated after accepting a stop token";
    const int32_t vocab_size = tokenizer_info_.GetVocabSize();
    int32_t* data = CheckAndGetBitmaskPtr(*next_token_bitmask, vocab_size, index);
    DynamicBitset bitset(vocab_size, reinterpret_cast<uint32_t*>(data));
    bitset.Reset();

    const std::vector<uint32_t>* cached = FindCachedMask(cursor_);
    std::vector<uint32_t> transient;
    if (cached == nullptr) {
      transient = ComputeMask(cursor_);
      cached = CacheMask(cursor_, std::move(transient));
    }
    std::memcpy(data, cached->data(), cached->size() * sizeof((*cached)[0]));
    if (IsCompleted()) {
      for (int32_t token_id : tokenizer_info_.GetStopTokenIds()) {
        if (token_id >= 0 && token_id < vocab_size) bitset.Set(token_id, true);
      }
    }
    return true;
  }

  void Rollback(int num_tokens) {
    XGRAMMAR_CHECK(num_tokens >= 0 && num_tokens <= static_cast<int>(history_.size()))
        << "Cannot roll back " << num_tokens << " steps from a history of " << history_.size();
    while (num_tokens-- > 0) {
      cursor_ = history_.back().cursor;
      terminated_ = history_.back().terminated;
      history_.pop_back();
    }
  }

  void Reset() {
    cursor_ = Cursor{};
    terminated_ = false;
    history_.clear();
  }

  bool IsCompleted() const {
    return cursor_.decode_state == JSONDecodeState::kNormal && cursor_.state != 0 &&
           states_[cursor_.state].occurrence_count == 1;
  }
  bool IsTerminated() const { return terminated_; }
  int32_t OccurrenceCount() const {
    return cursor_.state == 0 ? static_cast<int32_t>(source_size_ + 1)
                              : states_[cursor_.state].occurrence_count;
  }
  int32_t NumIndexStates() const { return static_cast<int32_t>(states_.size()); }

  std::size_t MemorySizeBytes() const {
    std::size_t result = sizeof(*this) + states_.capacity() * sizeof(State) +
                         history_.capacity() * sizeof(HistoryEntry);
    for (const auto& state : states_) {
      result += state.transitions.capacity() * sizeof(Transition);
    }
    for (const auto& [_, entry] : mask_cache_) {
      result += sizeof(int32_t) + sizeof(entry) + entry.mask.capacity() * sizeof(uint32_t);
    }
    result += mask_lru_.size() * sizeof(int32_t);
    result += transient_mask_.capacity() * sizeof(uint32_t);
    return result;
  }

 private:
  struct Transition {
    uint8_t byte;
    int32_t target;
  };
  struct State {
    int32_t length = 0;
    int32_t suffix_link = -1;
    int32_t occurrence_count = 0;
    std::vector<Transition> transitions;
  };
  enum class JSONDecodeState : uint8_t {
    kNormal,
    kEscape,
    kUnicode,
    kLowSurrogateBackslash,
    kLowSurrogateU,
  };
  struct Cursor {
    int32_t state = 0;
    JSONDecodeState decode_state = JSONDecodeState::kNormal;
    uint16_t unicode_value = 0;
    uint16_t high_surrogate = 0;
    uint8_t unicode_digits = 0;
  };
  struct HistoryEntry {
    Cursor cursor;
    bool terminated;
  };
  struct MaskCacheEntry {
    std::vector<uint32_t> mask;
    std::list<int32_t>::iterator lru_position;
  };

  static constexpr std::size_t kMaxMaskCacheBytes = 16 * 1024 * 1024;
  static constexpr std::size_t kMaxMaskCacheEntries = 4096;

  static int32_t FindTransition(const State& state, uint8_t byte) {
    for (const auto& transition : state.transitions) {
      if (transition.byte == byte) return transition.target;
    }
    return -1;
  }

  static void SetTransition(State* state, uint8_t byte, int32_t target) {
    for (auto& transition : state->transitions) {
      if (transition.byte == byte) {
        transition.target = target;
        return;
      }
    }
    state->transitions.push_back({byte, target});
  }

  bool WalkDecodedByte(Cursor* cursor, uint8_t byte) const {
    cursor->state = FindTransition(states_[cursor->state], byte);
    return cursor->state != -1;
  }

  bool WalkCodepoint(Cursor* cursor, uint32_t codepoint) const {
    for (uint8_t byte : CharToUTF8(codepoint)) {
      if (!WalkDecodedByte(cursor, byte)) return false;
    }
    return true;
  }

  bool FinishUnicodeEscape(Cursor* cursor) const {
    const uint16_t value = cursor->unicode_value;
    cursor->unicode_value = 0;
    cursor->unicode_digits = 0;
    if (cursor->high_surrogate != 0) {
      if (value < 0xdc00 || value > 0xdfff) return false;
      const uint32_t codepoint =
          0x10000 + ((cursor->high_surrogate - 0xd800) << 10) + (value - 0xdc00);
      cursor->high_surrogate = 0;
      cursor->decode_state = JSONDecodeState::kNormal;
      return WalkCodepoint(cursor, codepoint);
    }
    if (value >= 0xd800 && value <= 0xdbff) {
      cursor->high_surrogate = value;
      cursor->decode_state = JSONDecodeState::kLowSurrogateBackslash;
      return true;
    }
    if (value >= 0xdc00 && value <= 0xdfff) return false;
    cursor->decode_state = JSONDecodeState::kNormal;
    return WalkCodepoint(cursor, value);
  }

  bool WalkEncodedByte(Cursor* cursor, uint8_t byte) const {
    switch (cursor->decode_state) {
      case JSONDecodeState::kNormal:
        if (byte == '\\') {
          cursor->decode_state = JSONDecodeState::kEscape;
          return true;
        }
        if (byte == '"' || byte < 0x20) return false;
        return WalkDecodedByte(cursor, byte);
      case JSONDecodeState::kEscape:
        cursor->decode_state = JSONDecodeState::kNormal;
        switch (byte) {
          case '"':
          case '\\':
          case '/':
            return WalkDecodedByte(cursor, byte);
          case 'b':
            return WalkDecodedByte(cursor, '\b');
          case 'f':
            return WalkDecodedByte(cursor, '\f');
          case 'n':
            return WalkDecodedByte(cursor, '\n');
          case 'r':
            return WalkDecodedByte(cursor, '\r');
          case 't':
            return WalkDecodedByte(cursor, '\t');
          case 'u':
            cursor->decode_state = JSONDecodeState::kUnicode;
            return true;
          default:
            return false;
        }
      case JSONDecodeState::kUnicode: {
        int32_t value = HexCharToInt(static_cast<char>(byte));
        if (value == -1) return false;
        cursor->unicode_value = static_cast<uint16_t>((cursor->unicode_value << 4) | value);
        if (++cursor->unicode_digits == 4) return FinishUnicodeEscape(cursor);
        return true;
      }
      case JSONDecodeState::kLowSurrogateBackslash:
        if (byte != '\\') return false;
        cursor->decode_state = JSONDecodeState::kLowSurrogateU;
        return true;
      case JSONDecodeState::kLowSurrogateU:
        if (byte != 'u') return false;
        cursor->decode_state = JSONDecodeState::kUnicode;
        return true;
    }
    return false;
  }

  std::optional<Cursor> Walk(Cursor cursor, const std::string& encoded_bytes) const {
    for (uint8_t byte : encoded_bytes) {
      if (!WalkEncodedByte(&cursor, byte)) return std::nullopt;
    }
    return cursor;
  }

  std::vector<uint32_t> ComputeMask(const Cursor& cursor) const {
    const int32_t vocab_size = tokenizer_info_.GetVocabSize();
    std::vector<uint32_t> mask(static_cast<size_t>(GetBitmaskSize(vocab_size)), 0);
    DynamicBitset bitset(vocab_size, mask.data());
    const auto& vocab = tokenizer_info_.GetDecodedVocab();
    for (int32_t token_id = 0; token_id < vocab_size; ++token_id) {
      if (IsStopToken(token_id) || IsSpecialToken(token_id) ||
          token_id >= static_cast<int32_t>(vocab.size()) || vocab[token_id].empty()) {
        continue;
      }
      if (Walk(cursor, vocab[token_id]).has_value()) bitset.Set(token_id, true);
    }
    return mask;
  }

  const std::vector<uint32_t>* FindCachedMask(const Cursor& cursor) {
    if (cursor.decode_state != JSONDecodeState::kNormal) return nullptr;
    auto it = mask_cache_.find(cursor.state);
    if (it == mask_cache_.end()) return nullptr;
    mask_lru_.splice(mask_lru_.end(), mask_lru_, it->second.lru_position);
    return &it->second.mask;
  }

  const std::vector<uint32_t>* CacheMask(const Cursor& cursor, std::vector<uint32_t>&& mask) {
    if (cursor.decode_state != JSONDecodeState::kNormal ||
        mask.size() * sizeof(uint32_t) > kMaxMaskCacheBytes) {
      transient_mask_ = std::move(mask);
      return &transient_mask_;
    }
    const std::size_t bytes = mask.size() * sizeof(uint32_t);
    while (!mask_lru_.empty() && (mask_cache_bytes_ + bytes > kMaxMaskCacheBytes ||
                                  mask_cache_.size() >= kMaxMaskCacheEntries)) {
      const int32_t state = mask_lru_.front();
      mask_lru_.pop_front();
      mask_cache_bytes_ -= mask_cache_.at(state).mask.size() * sizeof(uint32_t);
      mask_cache_.erase(state);
    }
    mask_lru_.push_back(cursor.state);
    auto [it, inserted] = mask_cache_.emplace(
        cursor.state, MaskCacheEntry{std::move(mask), std::prev(mask_lru_.end())}
    );
    XGRAMMAR_DCHECK(inserted);
    mask_cache_bytes_ += bytes;
    return &it->second.mask;
  }

  bool IsStopToken(int32_t token_id) const {
    const auto& ids = tokenizer_info_.GetStopTokenIds();
    return std::find(ids.begin(), ids.end(), token_id) != ids.end();
  }

  bool IsSpecialToken(int32_t token_id) const {
    return std::find(special_token_ids_.begin(), special_token_ids_.end(), token_id) !=
           special_token_ids_.end();
  }

  void BuildIndex(const std::string& source) {
    source_size_ = source.size();
    states_.clear();
    states_.reserve(source.size() * 2 + 1);
    states_.push_back(State{});
    int32_t last = 0;
    for (uint8_t byte : source) {
      suffix_automata_detail::Extend(
          &states_,
          &last,
          byte,
          FindTransition,
          [](State* state, uint8_t symbol, int32_t target) { SetTransition(state, symbol, target); }
      );
    }
    suffix_automata_detail::PropagateOccurrenceCounts(&states_);
  }

  TokenizerInfo tokenizer_info_;
  std::vector<int32_t> special_token_ids_;
  std::vector<State> states_;
  std::unordered_map<int32_t, MaskCacheEntry> mask_cache_;
  std::list<int32_t> mask_lru_;
  std::vector<uint32_t> transient_mask_;
  std::size_t mask_cache_bytes_ = 0;
  std::vector<HistoryEntry> history_;
  std::size_t source_size_ = 0;
  Cursor cursor_;
  bool terminated_ = false;
};

UniqueSubstringMatcher::UniqueSubstringMatcher(
    const std::string& source, const TokenizerInfo& tokenizer_info
)
    : pimpl_(std::make_shared<Impl>(source, tokenizer_info)) {}

bool UniqueSubstringMatcher::AcceptToken(int32_t token_id) { return pimpl_->AcceptToken(token_id); }
bool UniqueSubstringMatcher::AcceptString(const std::string& input) {
  return pimpl_->AcceptString(input);
}
bool UniqueSubstringMatcher::FillNextTokenBitmask(DLTensor* bitmask, int index) {
  return pimpl_->FillNextTokenBitmask(bitmask, index);
}
void UniqueSubstringMatcher::Rollback(int num_tokens) { pimpl_->Rollback(num_tokens); }
void UniqueSubstringMatcher::Reset() { pimpl_->Reset(); }
bool UniqueSubstringMatcher::IsCompleted() const { return pimpl_->IsCompleted(); }
bool UniqueSubstringMatcher::IsTerminated() const { return pimpl_->IsTerminated(); }
int32_t UniqueSubstringMatcher::OccurrenceCount() const { return pimpl_->OccurrenceCount(); }
int32_t UniqueSubstringMatcher::NumIndexStates() const { return pimpl_->NumIndexStates(); }
std::size_t UniqueSubstringMatcher::MemorySizeBytes() const { return pimpl_->MemorySizeBytes(); }

}  // namespace xgrammar
