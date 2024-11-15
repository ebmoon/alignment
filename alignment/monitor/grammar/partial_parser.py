from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Self, Tuple

from lark.lexer import TerminalDef
from lark.parsers.lalr_analysis import (
    ParseTableBase,
    Shift,
    StateT as StateP
)

from .partial_lexer import PartialLexerFST

if TYPE_CHECKING:
    from interegular.fsm import State as StateL

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
                stack_updated = self._feed_terminals(stack + stack_push, terminals)
                if stack_updated:
                    parser_dest = stack_updated[-1]
                    if k == self.eos_token_id and parser_dest == self.end_state:
                        accepted[k] = (lexer_dest, stack_updated)
                    elif parser_dest not in self.invalid_pairs[lexer_dest]:
                        accepted[k] = (lexer_dest, stack_updated)
            else:
                # if there is no remaining terminals, the last action must be shift
                accepted[k] = (lexer_dest, stack + stack_push)

        return accepted

    def _feed_terminals(
            self,
            stack: Iterable[StateP],
            terminals: Iterable[str]
        ) -> Optional[Iterable[StateP]]:

        parser_state = stack[-1]
        parse_table = self.terminal_table

        for terminal in terminals:
            if parser_state not in parse_table.states:
                return None

            table_for_state = parse_table.states[parser_state]
            if terminal not in table_for_state:
                return None

            action, arg = table_for_state[terminal]
            if action is Shift:
                assert isinstance(arg, StateP)
                parser_state = arg
                stack.append(parser_state)
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

        Returns:
            `TokenParsingTable`: a token-level parse table
        """

        token_table = {}
        invalid_pairs = {}
        end_state = parse_table.end_states[start]

        # Build invalid pairs of lexer state and parser state
        for lexer_state in lexing_fst.states:
            invalid_pairs[lexer_state] = set()
            for parser_state in parse_table.states:
                valid_terminals = list(parse_table.states[parser_state].keys())
                reachable_terminals = lexing_fst.reachable_terminals[lexer_state]

                if len(set(valid_terminals) & set(reachable_terminals)) == 0:
                    invalid_pairs[lexer_state].add(parser_state)

        for lexer_state in lexing_fst.states:
            if lexer_state not in lexing_fst.map:
                continue

            transitions = lexing_fst.map[lexer_state]
            for parser_state in parse_table.states:
                token_transitions = _build_transitions(
                    parser_state, parse_table, transitions, 
                    eos_token_id, end_state, invalid_pairs)
                if len(token_transitions) > 0:
                    token_table[lexer_state, parser_state] = token_transitions

        return cls(token_table, parse_table, eos_token_id, start, invalid_pairs)

def _build_transitions(
        parser_state: StateP,
        parse_table: ParseTableBase,
        transitions: Dict[Any, Tuple["StateL", Iterable[str]]],
        eos_token_id: int,
        end_state: StateP,
        invalid_pairs: Dict["StateL", StateP]
    ) -> Dict[Any, Tuple["StateL", Iterable[StateP], Iterable[str]]]:

    token_transitions = {}
    for k, (lexer_dest, terminals) in transitions.items():
        if len(terminals) == 0:
            if parser_state not in invalid_pairs[lexer_dest]:
                token_transitions[k] = (lexer_dest, [], terminals)
        else:
            transition_result = _follow(parser_state, parse_table, terminals)
            if transition_result:
                stack, terminals = transition_result
                parser_dest = stack[-1]
                if k == eos_token_id and parser_dest == end_state:
                    token_transitions[k] = (lexer_dest, stack, terminals)
                elif len(terminals) > 0 or parser_dest not in invalid_pairs[lexer_dest]:
                    token_transitions[k] = (lexer_dest, stack, terminals)

    return token_transitions

def _follow(
        parser_state: StateP,
        parse_table: ParseTableBase, 
        terminals: Iterable[str]
    ) -> Optional[Tuple[Iterable[StateP], Iterable[str]]]:
    
    stack = []

    for i, terminal in enumerate(terminals):
        if parser_state not in parse_table.states:
            return None

        table_for_state = parse_table.states[parser_state]
        if terminal not in table_for_state:
            return None
        
        action, arg = table_for_state[terminal]
        if action is Shift:
            parser_state = arg
            stack.append(parser_state)
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

            else:   # can't be precomputed from here
                return stack, terminals[i:]
    
    return stack, []
