from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Iterable, Optional, Self, Set, Tuple

from lark.lexer import TerminalDef
from lark.parsers.lalr_analysis import (
    ParseTableBase,
    Shift,
    StateT as StateP
)

from collections import defaultdict

from .partial_lexer import PartialLexerFST

if TYPE_CHECKING:
    from interegular.fsm import State as StateL

class TerminalTrieNode:
    def __init__(self, node_id, parent=None):
        self.parent = parent
        self.node_id = node_id
        self.is_whole = False
        self.children = {}
        self.cache = {}

class TerminalTrie:
    """
    A trie of possible outputs of partial lexer FST
    """
    def __init__(self):
        self.root = TerminalTrieNode(0)
        self.count = 1

    def insert(self, terminals):
        node = self.root
        for terminal in terminals[:-1]:
            if terminal not in node.children:
                child_node = TerminalTrieNode(self.count)
                self.count += 1
                
                node.children[terminal] = child_node

            node = node.children[terminal]

        parent = node
        terminal = terminals[-1]
        if terminal not in node.children:
            child_node = TerminalTrieNode(self.count)
            self.count += 1
            
            node.children[terminal] = child_node

        node = node.children[terminal]
        node.parent = parent
        node.is_whole = True

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
            lexer: PartialLexerFST
        ):
        self.token_table = token_table
        self.terminal_table = terminal_table
        self.eos_token_id = eos_token_id
        self.start_state = terminal_table.start_states[start]
        self.end_state = terminal_table.end_states[start]
        self.lexer = lexer

    def acceptance(
            self,
            lexer_state: "StateL",
            stack: Iterable[StateP]
        ) -> Dict[int, Iterable[StateP]]:
        """
        Compute the set up accepted tokens from the current state, 
        and their corresponding next states after transition.
        
        Args:
            lexer_state(`StateL`): the current lexer state
            parser_state(`StateP`): the current parser state
            stack(`Iterable[StateP]`): the current stack of LR parser

        Returns:
            Dict[token_id, stack']:
                a map from allowed token_id to the state after transition
        """
        
        reachable_terminals = self.lexer.reachable_terminals

        parser_state = stack[-1]
        if self.token_table[lexer_state][parser_state] is None:
            return {}

        accepted = {}
        transitions = self.token_table[lexer_state][parser_state]
        for tokens, stack_push, terminals in transitions:
            if terminals:
                # Check if terminals can be consumed by the current stack
                stack_updated = self._feed_terminals(stack[:-1] + stack_push, terminals)
                if stack_updated:
                    parser_dest = stack_updated[-1]

                    if self.eos_token_id in tokens:
                        if parser_dest == self.end_state:
                            accepted[self.eos_token_id] = stack_updated
                    else:
                        for k in tokens:
                            accepted[k] = stack_updated

            else:
                # if there is no remaining terminals, the last action must be shift
                stack_updated = stack[:-1] + stack_push
                for k in tokens:
                    accepted[k] = stack_updated

        return accepted

    def _feed_terminals(
            self,
            stack: Iterable[StateP],
            terminals: Iterable[str]
        ) -> Optional[Iterable[StateP]]:

        parse_table = self.terminal_table

        stack = list(stack)

        for terminal in terminals[:-1]:
            if terminal in self.lexer.ignore_types:
                continue

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
                    stack.append(parser_state)
                    break
                else:   # Reduce
                    rule = arg
                    size = len(rule.expansion)

                    if size < len(stack):
                        if size:
                            del stack[-size:]

                        state = stack[-1]
                        nt_name = rule.origin.name
                        _action, parser_state = parse_table.states[state][nt_name]

                        assert _action is Shift
                        stack.append(parser_state)

                        if parser_state == self.end_state:
                            return stack

                    else:
                        return None

        prev_stack = stack
        terminal = terminals[-1]
        if terminal in self.lexer.ignore_types:
            return prev_stack

        while True:
            parser_state = stack[-1]
            if parser_state not in parse_table.states:
                return None

            table_for_state = parse_table.states[parser_state]
            if terminal not in table_for_state:
                return None

            action, arg = table_for_state[terminal]
            if action is Shift:
                return prev_stack
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

        return prev_stack

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

        end_state = parse_table.end_states[start]

        # Build prefix trie of possible terminal sequences
        trie = TerminalTrie()
        group_by_terminals = defaultdict(set)
        for src, transition in lexing_fst.map.items():
            for token_node_id, (dest, terminals) in transition.items():
                if token_node_id == lexing_fst.eos_node_id:
                    trie.insert(terminals)

                    # Store (src, dest) pairs for each terminal for later use
                    terminals_tuple = tuple(terminals)
                    group_by_terminals[src, terminals_tuple].add(token_node_id)                    

                else:
                    for reachable_terminal in lexing_fst.reachable_terminals[dest]:
                        extended_terminals = terminals + (reachable_terminal,)
                        trie.insert(extended_terminals)

                        # Store (src, dest) pairs for each terminal for later use
                        terminals_tuple = tuple(extended_terminals)
                        group_by_terminals[src, terminals_tuple].add(token_node_id)          

        # Update transition map for each terminal sequences
        id_map = {state:([state], []) for state in parse_table.states}
        trie.root.cache = id_map
        for terminal, child in trie.root.children.items():
            _compute_transition_dfs(
                lexing_fst.ignore_types, parse_table, end_state, child, terminal, id_map)

        # Build fused parse table
        token_table = [[[] for _ in parse_table.states] for _ in lexing_fst.states]

        for (lexer_src, terminals), node_ids in group_by_terminals.items():
            node = trie.traverse(terminals)
            tokens = {token for node_id in node_ids for token in lexing_fst.leaf_nodes[node_id].tokens}

            for parser_src, (stack, remainder) in node.cache.items():
                if remainder:
                    token_table[lexer_src][parser_src].append((tokens, stack, remainder))
                elif lexing_fst.eos_node_id in node_ids:
                    if stack[-1] == end_state:
                        token_table[lexer_src][parser_src].append((tokens, stack, remainder))
                else:
                    stack, remainder = node.parent.cache[parser_src]
                    token_table[lexer_src][parser_src].append((tokens, stack, remainder))

        del group_by_terminals
        del trie

        # Remove dummy lexer-parser state pairs
        dummy_pairs = [(lexer_src, parser_src) 
            for lexer_src in lexing_fst.states 
            for parser_src in parse_table.states
            if not token_table[lexer_src][parser_src]]

        for lexer_src, parser_src in dummy_pairs:
            token_table[lexer_src][parser_src] = None

        del dummy_pairs

        return cls(token_table, parse_table, eos_token_id, start, lexing_fst)

def _compute_transition_dfs(
    ignore_types: FrozenSet[str],
    parse_table: ParseTableBase,
    end_state: StateP,
    node: TerminalTrieNode, 
    terminal: str, 
    prev_result: Dict[StateP, Tuple[Iterable[StateP], List[str]]]
):
    """
    Update the map (src_state) -> (stack, remainder) for the current node.
    """

    result = {}
    for src, (stack, remainder) in prev_result.items():
        if remainder:
            result[src] = (stack, remainder + [terminal])
            continue

        if terminal in ignore_types:
            result[src] = (stack, [])
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
                result[src] = (stack + [parser_state], [])
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
                    result[src] = (stack, [terminal])
                    break

    for terminal, child in node.children.items():
        _compute_transition_dfs(
            ignore_types, parse_table, end_state, child, terminal, result)

    if node.is_whole or any(child.is_whole for child in node.children.values()):
        node.cache = result
    else:
        del result
