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

        vocabulary = {'0':0, '1':1, '01':2, 'EOS':3, '00':4, '11':5, '10':6}
        monitor = CFGMonitor(grammar_str, vocabulary, eos_token_id=3)

        state = monitor.state[0]
        print(state.acceptance)

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