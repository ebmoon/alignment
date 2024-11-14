from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Set, Self, Iterator, Iterable, Optional, Tuple

from lark.exceptions import UnexpectedCharacters
from lark.lexer import (
    BasicLexer,
    Lexer,
    LexerState,
    LexerThread,
    Token,
    TerminalDef
)

from interegular import FSM, parse_pattern
from interegular.fsm import Alphabet, OblivionError, State, TransitionKey, anything_else

if TYPE_CHECKING:
    from lark.lexer import LexerConf

class PartialLexer(Lexer):
    """
    A base class for partial lexer,
    that splits a prefix into a sequence of lexed tokens and unlexed suffix.
    """

    @abstractmethod
    def partial_lex(
        self, lexer_state: LexerState, parser_state: Any
    ) -> Tuple[Iterator[Token], Optional[str]]:
        """
        Split a prefix into a sequence of lexed tokens and unlexed suffix

        Args:
            lexer_state (`LexerState`): the current state of partial lexer
            parser_state (`Any`): an optional parser state for contextual lexer

        Returns:
            `Iterator[ParserToken]`: a pair of sequence of lexed tokens
            `Optional[str]`: unlexed suffix
        """

class PartialLexerThread(LexerThread):
    """
    A thread that ties a partial lexer instance and a lexer state, 
    to be used by the parser
    """

    def __init__(self, lexer: PartialLexer, lexer_state: LexerState):
        super().__init__(lexer, lexer_state)

    @classmethod
    def from_text(cls, lexer: PartialLexer, text: str) -> Self:
        return cls(lexer, LexerState(text))

    def partial_lex(
            self, parser_state: Any
        ) -> Tuple[Iterator[Token], Optional[str]]:
        """
        Split a prefix into a sequence of lexed tokens and unlexed suffix

        Args:
            parser_state (`Any`): an optional parser state for contextual lexer

        Returns:
            `Iterator[ParserToken]`: a pair of sequence of lexed tokens
            `Optional[str]`: unlexed suffix
        """
        return self.lexer.partial_lex(self.state, parser_state)

class PartialBasicLexer(BasicLexer, PartialLexer):
    """A non-contextual partial lexer"""

    def __init__(self, conf: "LexerConf"):
        super().__init__(conf)

    def partial_lex(
        self, lexer_state: LexerState, parser_state: Any = None
    ) -> Tuple[Iterator[Token], Optional[str]]:
        """
        Split a prefix into a sequence of lexed tokens and unlexed suffix

        Args:
            lexer_state (`PartialLexerState`): the current state of partial lexer
            parser_state (`Any`): an optional parser state (not used, for inheritance)

        Return:
            `Iterator[ParserToken]`: a pair of sequence of lexed tokens
            `Optional[str]`: unlexed suffix
        """
        lexer_tokens = []
        lexing_incomplete = False
        try:
            while lexer_state.line_ctr.char_pos < len(lexer_state.text):
                token = self.next_token(lexer_state)
                lexer_tokens.append(token)
        except UnexpectedCharacters:
            lexing_incomplete = True
        except EOFError:
            pass

        if lexing_incomplete:
            remainder = lexer_state.text[lexer_state.line_ctr.char_pos:]
            return lexer_tokens, remainder
        else:
            return lexer_tokens, None

class PartialLexerFST(BasicLexer):
    """
    A finite-state transducer implementation of partial lexer.
    """

    vocabulary: Dict[str, int]
    fsm: FSM
    initial: State
    states: Set[State]
    finals: Set[State]
    map: Dict[State, Dict[TransitionKey, Tuple[State, Iterable[TerminalDef]]]]
    final_map: Dict[State, TerminalDef]

    def __init__(self, conf: "LexerConf", vocabulary: Dict[str, int]):
        super().__init__(conf)

        self.vocabulary = vocabulary

        self.initial = None
        self.states = None
        self.finals = None
        self.fsm = None
        self.final_map = {}
        self._build_fsm()

        self.map = None
        self._build_map()

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
            final_map[state] = terminal_map[terminal_idx]

        self.fsm = fsm
        self.final_map = final_map
        self.initial = fsm.initial
        self.states = fsm.states
        self.finals = fsm.finals
    
    def _longest_match(
            self, state: State, lexeme: str
        ) -> Tuple[Optional[State], Optional[TerminalDef], str]:
        """
        Find the longest match of the input from the state.
        There are three possible cases:
            1. the input can be partially matched to a Terminal
            2. the transition stuck at a final state (i.e., matched to a terminal)
            3. the transition stuck at a non-final state (i.e., a prefix is matched to a terminal)
        We assume 1-lookahead lexer so the case 3 is discarded.

        Returns:
            Optional[State]: starting state after the longest match
            Optional[Token]: matched terminal token
            str: remainder after the longest match
        """

        alphabet = self.fsm.alphabet

        for i, symbol in enumerate(lexeme):
            if anything_else in alphabet and symbol not in alphabet:
                symbol = anything_else
            transition = alphabet[symbol]

            if not (state in self.fsm.map and transition in self.fsm.map[state]):
                if state in self.finals:
                    # Case 2: the transition stuck at a final stat
                    return self.initial, self.final_map[state], lexeme[i:]
                else:
                    # Case 3: the transition stuck at a non-final state
                    return None, None, ''

            state = self.fsm.map[state][transition]
        
        # Case 1: the input can be partially matched to a Terminal
        return state, None, ''

    def _compute_transition(
            self, state: State, token: str
        ) -> Optional[Tuple[State, Iterable[TerminalDef]]]:
        terminals = []
        while len(token) > 0:
            state, terminal, token = self._longest_match(state, token)
            if state is None:
                return None

            if terminal:
                terminals.append(terminal)
        return state, terminals

    def _build_map(self) -> Dict[State, Dict[TransitionKey, Tuple[State, Iterable[TerminalDef]]]]:
        fst_map = {state:{} for state in self.states}

        for state in self.states:
            for token, token_id in self.vocabulary.items():
                transition = self._compute_transition(state, token)
                if transition:
                    fst_map[state][token_id] = transition

        self.map = fst_map

    def follow(self, state: State, token_id: int) -> Optional[Tuple[State, Iterable[TerminalDef]]]:
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
    return _parallel(fsms)

def _parallel(fsms) -> Tuple[FSM, Dict[State, Dict[int, State]]]:
    """
        Crawl several FSMs in parallel, mapping the states of a larger meta-FSM.
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