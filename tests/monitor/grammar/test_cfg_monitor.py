from unittest import TestCase
from lark import Lark

from alignment.monitor.grammar.cfg_monitor import CFGMonitor

class CFTMonitorTest(TestCase):
    """Unit test for PartialLexerFST class"""

    def test_simple_grammar(self):
        grammar_str = """
            ?start: "00000"
                | "11111"
        """

        vocabulary = {0:'0', 1:'1', 2:'01', 3:'EOS', 4:'00', 5:'11', 6:'10'}
        monitor = CFGMonitor(grammar_str, vocabulary, eos_token_id=3)

        state = monitor.state[0]

        self.assertTrue(0 in state.acceptance)
        self.assertTrue(4 in state.acceptance)
        self.assertTrue(5 in state.acceptance)
        self.assertTrue(2 not in state.acceptance)
        self.assertTrue(3 not in state.acceptance)

        state = state.feed_token(4)

        self.assertTrue(0 in state.acceptance)
        self.assertTrue(4 in state.acceptance)
        self.assertTrue(5 not in state.acceptance)
        self.assertTrue(2 not in state.acceptance)
        self.assertTrue(3 not in state.acceptance)

        state = state.feed_token(4)

        self.assertTrue(0 in state.acceptance)
        self.assertTrue(4 not in state.acceptance)
        self.assertTrue(5 not in state.acceptance)
        self.assertTrue(2 not in state.acceptance)
        self.assertTrue(3 not in state.acceptance)

        state = state.feed_token(0)

        self.assertTrue(0 not in state.acceptance)
        self.assertTrue(4 not in state.acceptance)
        self.assertTrue(5 not in state.acceptance)
        self.assertTrue(2 not in state.acceptance)
        self.assertTrue(3 in state.acceptance)

    def test_json_grammar(self):
        grammar_str = """
            ?start: object

            ?object: "{\\"reasoning\\": " string_value ", \\"answer\\": " ans_value "}"

            ?string_value: "\\"" STRING "\\""

            ?ans_value: "\\"" ANSWER "\\""

            ANSWER: /\\([A-E]\\)/
            STRING.1: /[ \\t!#-\\[\\]-~]+/"""
        
        vocabulary = {
            0:'A', 1:'B', 2:'C', 3:'bb', 4:'"', 5:'reasoning', 6:'answer', 
            7:' ', 8:'EOS', 9:'{', 10:'}', 11:':', 12:'  ', 13:'\t', 14:',', 15:'(', 16:')'}
        monitor = CFGMonitor(grammar_str, vocabulary, eos_token_id=8)

        state = monitor.state[0]

        self.assertTrue(9 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(9)
        
        self.assertTrue(4 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(4)
        
        self.assertTrue(5 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(5)
        
        self.assertTrue(4 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(4)

        self.assertTrue(11 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(11)

        self.assertTrue(7 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(7)

        self.assertTrue(4 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(4)
        state = state.feed_token(3)
        state = state.feed_token(4)

        self.assertTrue(14 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(14)

        self.assertTrue(7 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(7)

        self.assertTrue(4 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(4)

        self.assertTrue(6 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(6)

        self.assertTrue(4 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(4)

        self.assertTrue(11 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(11)

        self.assertTrue(7 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(7)

        self.assertTrue(4 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(4)

        self.assertTrue(15 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(15)

        self.assertTrue(0 in state.acceptance)
        self.assertTrue(1 in state.acceptance)
        self.assertTrue(2 in state.acceptance)

        state = state.feed_token(2)

        self.assertTrue(16 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(16)

        self.assertTrue(4 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(4)

        self.assertTrue(10 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(10)

        self.assertTrue(8 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)

        state = state.feed_token(8)

    def test_dragon_book_grammar(self):
        grammar_str = """
            // Example 4.54 (Fig 4.41) from the Dragon Book
            ?start: ch ch
            ?ch: "c" ch | "d"
        """

        vocabulary = {0:'c', 1:'d', 2:'cc', 3:'cd', 4:'dc', 5:'dd', 6:'EOS'}
        monitor = CFGMonitor(grammar_str, vocabulary, eos_token_id=6)

        state = monitor.state[0]
        print(state.lexer_state, state.stack)
        print("")

        self.assertTrue(6 not in state.acceptance)
        self.assertTrue(len(state.acceptance) == 6)

        state = state.feed_token(4)
        print(state.lexer_state, state.stack)
        print("")

        self.assertTrue(0 in state.acceptance)
        self.assertTrue(1 in state.acceptance)
        self.assertTrue(2 in state.acceptance)
        self.assertTrue(3 in state.acceptance)
        self.assertTrue(4 not in state.acceptance)
        self.assertTrue(5 not in state.acceptance)
        self.assertTrue(6 not in state.acceptance)

        state = state.feed_token(3)

        print(state.lexer_state, state.stack)
        print("")

        print(state.parse_table.terminal_table.states)
        print("")

        print(state.lexer.map)
        print("")

        print(state.parse_table.token_table)
        print("")

        print(state.acceptance)

        self.assertTrue(6 in state.acceptance)
        self.assertTrue(len(state.acceptance) == 1)
