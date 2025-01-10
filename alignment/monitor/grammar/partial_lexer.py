from typing import TYPE_CHECKING, Dict, Set, List, Iterable, Optional, Tuple

from lark.lexer import (
    BasicLexer,
    TerminalDef,
    Pattern
)

from interegular import FSM, parse_pattern
from interegular.fsm import Alphabet, OblivionError, State, TransitionKey, anything_else

if TYPE_CHECKING:
    from lark.lexer import LexerConf

END_TERMINAL = '$END'

class TokenTrieNode:
    def __init__(self):
        self.children = {}
        self.cache = {}

class TokenTrie:
    """
    A trie of LLM vocabulary tokens
    """
    def __init__(self):
        self.root = TokenTrieNode()
    
    def insert(self, string):
        node = self.root
        for char in string:
            if char not in node.children:
                node.children[char] = TokenTrieNode()
            node = node.children[char]

    def traverse(self, string):
        node = self.root
        for char in string:
            node = node.children[char]
        
        return node

class PartialLexerFST(BasicLexer):
    """
    A finite-state transducer implementation of partial lexer.
    """

    vocabulary: Dict[int, str]
    fsm: FSM
    initial: State
    states: Set[State]
    finals: Set[State]
    map: Dict[State, Dict[TransitionKey, Tuple[State, Iterable[str]]]]
    final_map: Dict[State, str]
    reachable_terminals: Dict[State, str]

    def __init__(self, conf: "LexerConf", vocabulary: Dict[int, str], eos_token_id: int):
        super().__init__(conf)

        self.vocabulary = vocabulary
        self.eos_token_id = eos_token_id

        self.initial = None
        self.states = None
        self.finals = None
        self.fsm = None
        self.final_map = {}
        self._build_fsm()

        self.map = None
        self._build_map()

        self.reachables = None
        self._compute_reachable_terminals()

    def producible(self, terminals: Iterable[str]) -> Iterable[State]:
        """
        Compute a set of states that can produce one of target terminals

        Args:
            terminals(`Iterable[str]`): a set of target terminals 
        
        Return:
            `Iterable[State]`: Set of states that can produce one of terminals
        """
        finals = [state for state, terminal in self.final_map.items() \
                   if terminal in terminals]
        seen = set(finals)
        rev_reachable = finals.copy()

        i = 0
        while i < len(rev_reachable):
            current = rev_reachable[i]
            if current in self.map:
                for transition in self.map[current]:
                    next = self.map[current][transition]
                    if next not in seen:
                        rev_reachable.append(next)
                        seen.add(next)
            i += 1
        return False

    def _build_fsm(self):
        terminals = sorted(self.terminals, key=lambda t: t.priority)
        terminal_map = {i:t for i, t in enumerate(terminals)}
        regexps = [t.pattern.to_regexp() for t in terminal_map.values()]
        fsms = [parse_pattern(exp).to_fsm() for exp in regexps]

        fsm, final_state_map = _union(*fsms)

        final_map = {}
        for state in fsm.finals:
            # Assume lexer is not ambiguous (matched terminal is unique)
            terminal_idx = final_state_map[state]
            final_map[state] = terminal_map[terminal_idx].name

        self.fsm = fsm
        self.final_map = final_map
        self.initial = fsm.initial
        self.states = fsm.states
        self.finals = fsm.finals

    def _compute_transition_dfs(
        self, node: TokenTrieNode, char: str, prev_result: Dict[State, Tuple[State, List[str]]]
    ) -> Dict[State, Tuple[State, List[str]]]:
        """
        Update the map (src_state) -> (dest_state, output_terminals) for the current node.

        The map is based on the longest match of the input from the state.
        There are three possible cases:
            1. the input can be partially matched to a Terminal
            2. the transition stuck at a final state (i.e., matched to a terminal)
            3. the transition stuck at a non-final state (i.e., a prefix is matched to a terminal)
        We assume 1-lookahead lexer so the case 3 is discarded.
        """
        result = {}

        for src, (dest, out) in prev_result.items():
            symbol = char
            if anything_else in self.fsm.alphabet and symbol not in self.fsm.alphabet:
                symbol = anything_else
            transition = self.fsm.alphabet[symbol]

            if not (dest in self.fsm.map and transition in self.fsm.map[dest]):
                if dest in self.finals and transition in self.fsm.map[self.initial]:
                    # Case 2: the transition stuck at a final stat
                    result[src] = (
                        self.fsm.map[self.initial][transition], 
                        out + [self.final_map[dest]])
                else:
                    # Case 3: the transition stuck at a non-final state
                    continue
        
        node.cache = result

        for char, child in node.children.items():
            self._compute_transition_dfs(child, char, result)

        return result

    def _build_map(self):
        # Build prefix trie of vocabulary tokens
        trie = TokenTrie()
        for token_id, token in self.vocabulary.items():
            if token_id != self.eos_token_id:
                trie.insert(token)

        # Update transition map
        id_map = {state:(state, []) for state in self.states}
        for char, child in trie.root.children.items():
            self._compute_transition_dfs(child, char, id_map)

        # Build FST map
        fst_map = {state:{} for state in self.states}
        for token_id, token in self.vocabulary.items():
            if token_id == self.eos_token_id:
                for state in self.finals:
                    fst_map[state][token_id] = (self.initial, (self.final_map[state], END_TERMINAL))
            else:
                cached_result = trie.traverse(token).cache
                for src, (dest, out) in cached_result.items():
                    fst_map[src][token] = (dest, out)

        del trie
        self.map = fst_map

    def _compute_reachable_terminals_single(self, state: State) -> Iterable[str]:
        # TODO: Avoid repetitive computation
        seen = {state}
        reachable = [state]
        terminals = set()
        i = 0
        while i < len(reachable):
            current = reachable[i]
            if current in self.finals:
                terminals.add(self.final_map[current])
            if current in self.fsm.map:
                for transition in self.fsm.map[current]:
                    next_state = self.fsm.map[current][transition]
                    if next_state not in seen:
                        reachable.append(next_state)
                        seen.add(next_state)
            i += 1

        return terminals

    def _compute_reachable_terminals(self):
        reachable_terminals = {}

        for state in self.states:
            reachable_terminals[state] = self._compute_reachable_terminals_single(state)

        self.reachable_terminals = reachable_terminals

    def follow(self, state: State, token_id: int) -> Optional[Tuple[State, Iterable[str]]]:
        """
        Feed a token from a source state,
        return the destination state and corresponding output 

        Args:
            state (`State`): a source state
            token_id (`int`): the index of token in the vocabulary
        
        Returns:
            `State`: destination state
            `Iterable[TerminalDef]`: lexed tokens
        """
        if not (state in self.map and token_id in self.map[state]):
            return None

        return self.map[state][token_id]

# These methods are modified from the implementation of interegular package:
# https://github.com/MegaIng/interegular

def _union(*fsms: FSM) -> Tuple[FSM, Dict[State, Dict[int, State]]]:
    """
        Union several FSMs, mapping the states of a larger meta-FSM.
        To determine whether a state in the larger FSM is final.
    """
    alphabet, new_to_old = Alphabet.union(*[fsm.alphabet for fsm in fsms])

    initial = {i: fsm.initial for (i, fsm) in enumerate(fsms)}

    # dedicated function accepts a "superset" and returns the next "superset"
    # obtained by following this transition in the new FSM
    def follow(current, new_transition, fsm_range=tuple(enumerate(fsms))):
        next_map = {}
        for i, f in fsm_range:
            old_transition = new_to_old[i][new_transition]
            if i in current \
                    and current[i] in f.map \
                    and old_transition in f.map[current[i]]:
                next_map[i] = f.map[current[i]][old_transition]
        if not next_map:
            raise OblivionError
        return next_map

    # Determine the "is final?" condition of each substate, then pass it to the
    # test to determine finality of the overall FSM.
    def final(state, fsm_range=tuple(enumerate(fsms))):
        accepts = [i in state and state[i] in fsm.finals for (i, fsm) in fsm_range]
        accepts_fsm = [i for (i, fsm) in fsm_range if i in state and state[i] in fsm.finals]
        return any(accepts), accepts_fsm

    return _crawl(alphabet, initial, final, follow)


def _crawl(alphabet, initial, final, follow) -> Tuple[FSM, Dict[State, int]]:
    """
        Given the above conditions and instructions, crawl a new unknown FSM,
        mapping its states, final states and transitions. Return the new FSM.
        This is a pretty powerful procedure which could potentially go on
        forever if you supply an evil version of follow().
    """

    states = [initial]
    finals = set()
    fsm_map = {}

    final_map = {}

    # iterate over a growing list
    i = 0
    while i < len(states):
        state = states[i]

        # add to finals
        is_final, fsm_idx = final(state)
        if is_final:
            finals.add(i)
            final_map[i] = fsm_idx[0]

        # compute map for this state
        fsm_map[i] = {}
        for transition in alphabet.by_transition:
            try:
                next_map = follow(state, transition)
            except OblivionError:
                # Reached an oblivion state. Don't list it.
                continue
            else:
                try:
                    j = states.index(next_map)
                except ValueError:
                    j = len(states)
                    states.append(next_map)
                fsm_map[i][transition] = j

        i += 1

    return FSM(
        alphabet=alphabet,
        states=range(len(states)),
        initial=0,
        finals=finals,
        map=fsm_map,
        __no_validation__=True,
    ), final_map
