/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/suffix_automata.h
 * \brief Suffix automaton construction for substring expressions.
 */
#ifndef XGRAMMAR_SUFFIX_AUTOMATA_H_
#define XGRAMMAR_SUFFIX_AUTOMATA_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "fsm.h"

namespace xgrammar {

namespace suffix_automata_detail {

// Static grammars use string symbols and ordered maps; the runtime matcher uses byte symbols and
// compact vectors. This shares the algorithm while allowing each use to keep suitable storage.
template <typename State, typename Symbol, typename FindTransition, typename SetTransition>
void Extend(
    std::vector<State>* states,
    int32_t* last,
    const Symbol& symbol,
    FindTransition find_transition,
    SetTransition set_transition
) {
  const int32_t current = static_cast<int32_t>(states->size());
  states->push_back(State{});
  (*states)[current].length = (*states)[*last].length + 1;
  (*states)[current].occurrence_count = 1;

  int32_t parent = *last;
  while (parent != -1 && find_transition((*states)[parent], symbol) == -1) {
    set_transition(&(*states)[parent], symbol, current);
    parent = (*states)[parent].suffix_link;
  }
  if (parent == -1) {
    (*states)[current].suffix_link = 0;
  } else {
    const int32_t target = find_transition((*states)[parent], symbol);
    if ((*states)[parent].length + 1 == (*states)[target].length) {
      (*states)[current].suffix_link = target;
    } else {
      const int32_t clone = static_cast<int32_t>(states->size());
      states->push_back((*states)[target]);
      (*states)[clone].length = (*states)[parent].length + 1;
      (*states)[clone].occurrence_count = 0;
      while (parent != -1 && find_transition((*states)[parent], symbol) == target) {
        set_transition(&(*states)[parent], symbol, clone);
        parent = (*states)[parent].suffix_link;
      }
      (*states)[target].suffix_link = clone;
      (*states)[current].suffix_link = clone;
    }
  }
  *last = current;
}

template <typename State>
void PropagateOccurrenceCounts(std::vector<State>* states) {
  std::vector<int32_t> by_length(states->size());
  for (int32_t i = 0; i < static_cast<int32_t>(states->size()); ++i) by_length[i] = i;
  std::sort(by_length.begin(), by_length.end(), [&](int32_t lhs, int32_t rhs) {
    return (*states)[lhs].length > (*states)[rhs].length;
  });
  for (int32_t state : by_length) {
    const int32_t suffix_link = (*states)[state].suffix_link;
    if (suffix_link != -1) {
      (*states)[suffix_link].occurrence_count += (*states)[state].occurrence_count;
    }
  }
}

}  // namespace suffix_automata_detail

/*!
 * \brief Builds the automaton of a substring expression via a chunk-level suffix automaton.
 */
class SuffixAutomata {
 public:
  /*!
   * \brief Build an FSM that accepts exactly the contiguous subsequences of the chunk list,
   * including the empty one.
   * \details A suffix automaton is built over the chunk sequence (each chunk is one symbol), so
   * the number of automaton states grows linearly with the number of chunks. Every automaton
   * state is accepting. Each chunk-labeled transition is then expanded into a chain of byte
   * transitions; an empty chunk becomes an epsilon transition.
   * \param chunks The list of byte string chunks. Chunks may be empty or repeated.
   * \param unique_only If true, accept only non-empty contiguous subsequences that occur
   * exactly once in the chunk sequence. Occurrences may overlap.
   * \return The FSM with start and end states.
   */
  static FSMWithStartEnd Build(const std::vector<std::string>& chunks, bool unique_only = false);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUFFIX_AUTOMATA_H_
