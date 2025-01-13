from typing import TYPE_CHECKING, Any, Dict, List, Iterable, Optional, Self, Tuple

from lark.lexer import TerminalDef
from lark.parsers.lalr_analysis import (
    ParseTableBase,
    Shift,
    StateT as StateP
)

from .partial_lexer import PartialLexerFST

if TYPE_CHECKING:
    from interegular.fsm import State as StateL

class TerminalTrieNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}
        self.cache = {}

class TerminalTrie:
    """
    A trie of possible outputs of partial lexer FST
    """
    def __init__(self):
        self.root = TerminalTrieNode()

    def insert(self, terminals):
        node = self.root
        for terminal in terminals:
            if terminal not in node.children:
                node.children[terminal] = TerminalTrieNode(node)
            node = node.children[terminal]

    def traverse(self, terminals):
        node = self.root
        for terminal in terminals:
            node = node.children[terminal]
        
        return node

class TokenParsingTable:
    """
    A LLM token level parsing table
    """

    def __init__(
            self,
            token_table,
            terminal_table: ParseTableBase,
            eos_token_id: int,
            start: str,
            valid_pairs: Dict["StateL", StateP],
            reachable_terminals: Dict["StateL", Iterable[str]]
        ):
        self.token_table = token_table
        self.terminal_table = terminal_table
        self.eos_token_id = eos_token_id
        self.start_state = terminal_table.start_states[start]
        self.end_state = terminal_table.end_states[start]
        self.valid_pairs = valid_pairs
        self.reachable_terminals = reachable_terminals

    def acceptance(
            self,
            lexer_state: "StateL",
            stack: Iterable[StateP]
        ) -> Dict[int, Tuple["StateL", Iterable[StateP]]]:
        """
        Compute the set up accepted tokens from the current state, 
        and their corresponding next states after transition.
        
        Args:
            lexer_state(`StateL`): the current lexer state
            parser_state(`StateP`): the current parser state
            stack(`Iterable[StateP]`): the current stack of LR parser

        Returns:
            Dict[token_id, (lexer_state', parser_state', stack')]:
                a map from allowed token_id to the state after transition
        """
        
        parser_state = stack[-1]
        if (lexer_state, parser_state) not in self.token_table:
            return {}

        accepted = {}
        transitions = self.token_table[lexer_state, parser_state]
        for tokens, lexer_dest, stack_push, terminals in transitions:
            if len(terminals) > 0:
                # Check if terminals can be consumed by the current stack
                stack_updated = self._feed_terminals(stack[:-1] + stack_push, terminals)
                if stack_updated:
                    parser_dest = stack_updated[-1]

                    for k in tokens:
                        if k == self.eos_token_id and parser_dest == self.end_state:
                            accepted[k] = (lexer_dest, stack_updated)
                        else:
                            for reachable_terminal in self.reachable_terminals[lexer_dest]:
                                acceptable = self._feed_terminals(stack_updated, [reachable_terminal])
                                if acceptable:
                                    accepted[k] = (lexer_dest, stack_updated)
                                    break
            else:
                # if there is no remaining terminals, the last action must be shift
                for k in tokens:
                    stack_updated = stack[:-1] + stack_push
                    parser_dest = stack_updated[-1]
                    if k == self.eos_token_id and parser_dest == self.end_state:
                        accepted[k] = (lexer_dest, stack_updated)
                    else:
                        for reachable_terminal in self.reachable_terminals[lexer_dest]:
                            acceptable = self._feed_terminals(stack_updated, [reachable_terminal])
                            if acceptable:
                                accepted[k] = (lexer_dest, stack_updated)
                                break

        return accepted

    def _feed_terminals(
            self,
            stack: Iterable[StateP],
            terminals: Iterable[str]
        ) -> Optional[Iterable[StateP]]:

        parse_table = self.terminal_table

        for terminal in terminals:
            while True:
                parser_state = stack[-1]
                if parser_state not in parse_table.states:
                    return None

                table_for_state = parse_table.states[parser_state]
                if terminal not in table_for_state:
                    return None

                action, arg = table_for_state[terminal]
                if action is Shift:
                    parser_state = arg
                    stack = stack + [parser_state]
                    break
                else:   # Reduce
                    rule = arg
                    size = len(rule.expansion)

                    if size < len(stack):
                        if size:
                            stack = stack[:-size]

                        state = stack[-1]
                        nt_name = rule.origin.name
                        _action, parser_state = parse_table.states[state][nt_name]

                        assert _action is Shift
                        stack = stack + [parser_state]

                        if parser_state == self.end_state:
                            return stack

                    else:
                        return None

        return stack

    @classmethod
    def from_terminal_parse_table(
        cls,
        lexing_fst: PartialLexerFST,
        parse_table: ParseTableBase,
        eos_token_id: int,
        start: str
    ) -> Self:
        """
        Construct a LLM token-level parse table from terminal-level parse table.

        Args:
            lexing_fst(`PartialLexerFST`): a token-to-terminal lexing transducer
            parse_table(`ParseTableBase`): a terminal-level parse table
            eos_token_id(`int`): the index of eos token in vocabulary

        Returns:
            `TokenParsingTable`: a token-level parse table
        """

        valid_pairs = {}
        end_state = parse_table.end_states[start]

        # Build invalid pairs of lexer state and parser state
        for lexer_state in lexing_fst.states:
            valid_pairs[lexer_state] = set()
            for parser_state in parse_table.states:
                valid_terminals = set(parse_table.states[parser_state].keys())
                reachable_terminals = lexing_fst.reachable_terminals[lexer_state]

                if len(valid_terminals & reachable_terminals) > 0:
                    valid_pairs[lexer_state].add(parser_state)

        # Build prefix trie of possible terminal sequences
        trie = TerminalTrie()
        group_by_terminals = {}
        for src, transition in lexing_fst.map.items():
            for token_id, (dest, terminals) in transition.items():
                if token_id == eos_token_id:
                    trie.insert(terminals)

                    # Store (src, dest) pairs for each terminal for later use
                    terminals_tuple = tuple(terminals)
                    if terminals_tuple not in group_by_terminals:
                        group_by_terminals[terminals_tuple] = {}
                    if (src, dest) not in group_by_terminals[terminals_tuple]:
                        group_by_terminals[terminals_tuple][src, dest] = set()
                    group_by_terminals[terminals_tuple][src, dest].add(token_id)                    

                    continue

                for reachable_terminal in lexing_fst.reachable_terminals[dest]:
                    extended_terminals = terminals + (reachable_terminal,)
                    trie.insert(extended_terminals)

                    # Store (src, dest) pairs for each terminal for later use
                    terminals_tuple = tuple(extended_terminals)
                    if terminals_tuple not in group_by_terminals:
                        group_by_terminals[terminals_tuple] = {}
                    if (src, dest) not in group_by_terminals[terminals_tuple]:
                        group_by_terminals[terminals_tuple][src, dest] = set()
                    group_by_terminals[terminals_tuple][src, dest].add(token_id)

        import time
        start_t = time.time()

        # Update transition map for each terminal sequences
        id_map = {state:([state], []) for state in parse_table.states}
        trie.root.cache = id_map
        for terminal, child in trie.root.children.items():
            _compute_transition_dfs(parse_table, end_state, child, terminal, id_map)

        end_t = time.time()
        print(f"Precomputation for terminal trie: {end_t - start_t} s")

        # Build fused parse table
        token_table = {
            (lexer_state, parser_state):[] 
            for lexer_state in lexing_fst.states 
            for parser_state in parse_table.states}

        start_t = time.time()

        for terminals, lexer_transitions in group_by_terminals.items():
            node = trie.traverse(terminals)

            for (lexer_src, lexer_dest), tokens in lexer_transitions.items():
                for parser_src in valid_pairs[lexer_src]:
                    if parser_src not in node.cache:
                        continue

                    if eos_token_id in tokens:
                        stack, remainder = node.cache[parser_src]
                        parser_dest = stack[-1]

                        if len(remainder) > 0 or parser_dest == end_state:
                            token_table[lexer_src, parser_src].append((tokens, lexer_dest, stack, remainder))
                    
                    else:
                        stack, remainder = node.parent.cache[parser_src]
                        parser_dest = stack[-1]

                        token_table[lexer_src, parser_src].append((tokens, lexer_dest, stack, remainder))

                    

        end_t = time.time()
        print(f"Parser table construction: {end_t - start_t} s")

        # Remove dummy lexer-parser state pairs
        dummy_pairs = []
        for (lexer_src, parser_src), d in token_table.items():
            if len(d) == 0:
                dummy_pairs.append((lexer_src, parser_src))
        
        for lexer_src, parser_src in dummy_pairs:
            del token_table[lexer_src, parser_src]

        return cls(token_table, parse_table, eos_token_id, start, valid_pairs, lexing_fst.reachable_terminals)

def _compute_transition_dfs(
    parse_table: ParseTableBase,
    end_state: StateP,
    node: TerminalTrieNode, 
    terminal: str, 
    prev_result: Dict[StateP, Tuple[Iterable[StateP], List[str]]]
) -> Dict[StateP, Tuple[Iterable[StateP], List[str]]]:
    """
    Update the map (src_state) -> (stack, remainder) for the current node.
    """
    result = {}

    for src, (stack, remainder) in prev_result.items():
        if len(remainder) > 0:
            result[src] = (stack, remainder + [terminal])
            continue

        # Follow parser table as long as possible
        while True:
            parser_state = stack[-1]
            if parser_state not in parse_table.states:
                break

            table_for_state = parse_table.states[parser_state]
            if terminal not in table_for_state:
                break
            
            action, arg = table_for_state[terminal]
            if action is Shift:
                parser_state = arg
                result[src] = (stack + [parser_state], remainder)
                break

            else:   # Reduce
                rule = arg
                size = len(rule.expansion)

                if size < len(stack):
                    if size:
                        stack = stack[:-size]

                    state = stack[-1]
                    nt_name = rule.origin.name
                    _action, parser_state = parse_table.states[state][nt_name]
                
                    assert _action is Shift
                    stack = stack + [parser_state]

                    if parser_state == end_state:
                        result[src] = (stack, [])
                        break

                else:   # can't be precomputed from here
                    result[src] = (stack, remainder + [terminal])
                    break

    node.cache = result

    for char, child in node.children.items():
        _compute_transition_dfs(parse_table, end_state, child, char, result)

    return result