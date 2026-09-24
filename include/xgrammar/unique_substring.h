/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/unique_substring.h
 * \brief Runtime-bound unique-substring constraint for transient source text.
 */
#ifndef XGRAMMAR_UNIQUE_SUBSTRING_H_
#define XGRAMMAR_UNIQUE_SUBSTRING_H_

#include <dlpack/dlpack.h>
#include <xgrammar/object.h>
#include <xgrammar/tokenizer_info.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace xgrammar {

/*!
 * \brief Match a non-empty substring that occurs exactly once in runtime-bound source bytes.
 *
 * \details This prototype is intended for transient constraints such as the `old_str` argument
 * of a coding agent's search-and-replace tool. Construction builds a suffix automaton directly
 * from the current file contents. Token bytes are decoded as JSON string content before matching,
 * so escaped quotes, backslashes, control characters, and Unicode escapes match their source
 * bytes. Continuing tokens are allowed while at least one occurrence remains; a tokenizer stop
 * token is allowed only when exactly one occurrence remains and no escape is incomplete.
 * Occurrences may overlap.
 *
 * This matcher is deliberately independent of GrammarCompiler: changing the source rebuilds only
 * the suffix index and does not compile the source text into the surrounding tool-call grammar.
 */
class UniqueSubstringMatcher {
 public:
  UniqueSubstringMatcher(const std::string& source, const TokenizerInfo& tokenizer_info);

  bool AcceptToken(int32_t token_id);
  bool AcceptString(const std::string& input);
  bool FillNextTokenBitmask(DLTensor* next_token_bitmask, int index = 0);
  void Rollback(int num_tokens = 1);
  void Reset();

  bool IsCompleted() const;
  bool IsTerminated() const;
  int32_t OccurrenceCount() const;
  int32_t NumIndexStates() const;
  std::size_t MemorySizeBytes() const;

  XGRAMMAR_DEFINE_PIMPL_METHODS(UniqueSubstringMatcher);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_UNIQUE_SUBSTRING_H_
