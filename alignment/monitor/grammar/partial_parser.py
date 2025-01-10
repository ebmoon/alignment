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
    def __init__(self):
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
                node.children[terminal] = TerminalTrieNode()
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
            invalid_pairs: Dict["StateL", StateP]
        ):
        self.token_table = token_table
        self.terminal_table = terminal_table
        self.eos_token_id = eos_token_id
        self.start_state = terminal_table.start_states[start]
        self.end_state = terminal_table.end_states[start]
        self.invalid_pairs = invalid_pairs

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
        for k, (lexer_dest, stack_push, terminals) in transitions.items():
            if len(terminals) > 0:
                # Check if terminals can be consumed by the current stack
                stack_updated = self._feed_terminals(stack[:-1] + stack_push, terminals)
                if stack_updated:
                    parser_dest = stack_updated[-1]
                    if k == self.eos_token_id and parser_dest == self.end_state:
                        accepted[k] = (lexer_dest, stack_updated)
                    elif parser_dest not in self.invalid_pairs[lexer_dest]:
                        accepted[k] = (lexer_dest, stack_updated)
            else:
                # if there is no remaining terminals, the last action must be shift
                accepted[k] = (lexer_dest, stack[:-1] + stack_push)

        return accepted

    def _feed_terminals(
            self,
            stack: Iterable[StateP],
            terminals: Iterable[str]
        ) -> Optional[Iterable[StateP]]:

        parser_state = stack[-1]
        parse_table = self.terminal_table

        for terminal in terminals:
            while True:
                if parser_state not in parse_table.states:
                    return None

                table_for_state = parse_table.states[parser_state]
                if terminal not in table_for_state:
                    return None

                action, arg = table_for_state[terminal]
                if action is Shift:
                    parser_state = arg
                    stack.append(parser_state)
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
                        stack.append(parser_state)

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

        invalid_pairs = {}
        end_state = parse_table.end_states[start]

        # Build invalid pairs of lexer state and parser state
        for lexer_state in lexing_fst.states:
            invalid_pairs[lexer_state] = set()
            for parser_state in parse_table.states:
                valid_terminals = list(parse_table.states[parser_state].keys())
                reachable_terminals = lexing_fst.reachable_terminals[lexer_state]

                if len(set(valid_terminals) & reachable_terminals) == 0:
                    invalid_pairs[lexer_state].add(parser_state)

        # Build prefix trie of possible terminal sequences
        trie = TerminalTrie()
        group_by_terminals = {}
        for src, transition in lexing_fst.map.items():
            for token_id, (dest, terminals) in transition.items():
                trie.insert(terminals)

                # Store (src, dest) pairs for each terminal for later use
                terminals_tuple = tuple(terminals)
                if terminals_tuple not in group_by_terminals:
                    group_by_terminals[terminals_tuple] = []
                group_by_terminals[terminals_tuple].append((token_id, src, dest))

        # Update transition map for each terminal sequences
        id_map = {state:([state], []) for state in parse_table.states}
        trie.root.cache = id_map
        for terminal, child in trie.root.children.items():
            _compute_transition_dfs(parse_table, end_state, child, terminal, id_map)

        # Build fused parse table
        token_table = {
            (lexer_state, parser_state):{} 
            for lexer_state in lexing_fst.states 
            for parser_state in parse_table.states}

        for terminals, lexer_transitions in group_by_terminals.items():
            parser_transitions = trie.traverse(terminals).cache

            for token_id, lexer_src, lexer_dest in lexer_transitions:
                for parser_src, (stack, remainder) in parser_transitions.items():
                    parser_dest = stack[-1]
                    if token_id == eos_token_id and parser_dest == end_state:
                        token_table[lexer_src, parser_src][token_id] = (lexer_dest, stack, remainder)
                    elif len(remainder) > 0 or parser_dest not in invalid_pairs[lexer_dest]:
                        token_table[lexer_src, parser_src][token_id] = (lexer_dest, stack, remainder)

        # Remove dummy lexer-parser state pairs
        dummy_pairs = []
        for (lexer_src, parser_src), d in token_table.items():
            if len(d) == 0:
                dummy_pairs.append((lexer_src, parser_src))
        
        for lexer_src, parser_src in dummy_pairs:
            del token_table[lexer_src, parser_src]

        return cls(token_table, parse_table, eos_token_id, start, invalid_pairs)

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
        parser_state = stack[-1]
        while True:
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
                        result[srt] = (stack, [])
                        break

                else:   # can't be precomputed from here
                    result[src] = (stack, remainder + [terminal])
                    break

    node.cache = result

    for char, child in node.children.items():
        _compute_transition_dfs(parse_table, end_state, child, char, result)

    return result