/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/suffix_automata.cc
 */

#include "suffix_automata.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace xgrammar {

FSMWithStartEnd SuffixAutomata::Build(const std::vector<std::string>& chunks, bool unique_only) {
  // Step 1. Build the suffix automaton over the chunk sequence with the standard online
  // construction. Each chunk is treated as one symbol of the alphabet.
  struct State {
    int32_t length = 0;
    int32_t suffix_link = -1;
    std::map<std::string, int32_t> transitions;
    int32_t occurrence_count = 0;
  };

  std::vector<State> states(1);
  int32_t last = 0;
  for (const std::string& chunk : chunks) {
    suffix_automata_detail::Extend(
        &states,
        &last,
        chunk,
        [](const State& state, const std::string& symbol) {
          auto it = state.transitions.find(symbol);
          return it == state.transitions.end() ? -1 : it->second;
        },
        [](State* state, const std::string& symbol, int32_t target) {
          state->transitions[symbol] = target;
        }
    );
  }

  // Propagate terminal contributions through suffix links to obtain each state's end-position
  // count. All substrings represented by one state have the same occurrence count.
  if (unique_only) {
    suffix_automata_detail::PropagateOccurrenceCounts(&states);
  }

  // Step 2. Expand the chunk-level automaton into a byte-level FSM. Automaton state i maps to
  // FSM state i; every automaton state is accepting. A chunk-labeled transition becomes a chain
  // of byte transitions through fresh intermediate states; an empty chunk becomes an epsilon
  // transition.
  FSM fsm(static_cast<int>(states.size()));
  std::vector<int32_t> end_states;
  end_states.reserve(states.size());
  for (int32_t index = 0; index < static_cast<int32_t>(states.size()); ++index) {
    if (!unique_only || (index != 0 && states[index].occurrence_count == 1)) {
      end_states.push_back(index);
    }
  }
  for (int32_t index = 0; index < static_cast<int32_t>(states.size()); ++index) {
    for (const auto& [chunk, target] : states[index].transitions) {
      if (chunk.empty()) {
        fsm.AddEpsilonEdge(index, target);
        continue;
      }
      int current_state = index;
      for (size_t offset = 0; offset < chunk.size(); ++offset) {
        int next_state = offset + 1 == chunk.size() ? static_cast<int>(target) : fsm.AddState();
        uint8_t byte = static_cast<uint8_t>(chunk[offset]);
        fsm.AddEdge(current_state, next_state, byte, byte);
        current_state = next_state;
      }
    }
  }
  return FSMWithStartEnd(fsm, 0, std::move(end_states));
}

}  // namespace xgrammar
