from unittest import TestCase
from lark import Lark

from alignment.monitor.grammar.partial_lexer import PartialLexerFST

class PartialLexerFSTTest(TestCase):
    """Unit test for PartialLexerFST class"""

    def test_calc_grammar(self):
        calc_grammar = """
            ?start: sum
                | NAME "=" sum    

            ?sum: product
                | sum "+" product   
                | sum "-" product   
                | sum "---" product 

            ?product: atom
                | product "*" atom  
                | product "/" atom 

            ?atom: NUMBER           
                | "-" atom        
                | NAME            
                | "(" sum ")"

            %import common.CNAME -> NAME
            %import common.NUMBER
            %import common.WS_INLINE

            %ignore WS_INLINE
        """

        lexer_conf = Lark(calc_grammar, parser='lalr').lexer_conf
        vocabulary = {'---':1, '-':3, 'aa':4, 'a':5, 'abb-':6, ' ':7, 'a ':8, '---a':9}

        fst = PartialLexerFST(lexer_conf, vocabulary)

        state_1, out = fst.follow(fst.initial, vocabulary['-'])
        self.assertEqual(len(out), 0)

        _, out = fst.follow(state_1, vocabulary['---'])
        self.assertEqual(len(out), 1)

        state_1, out = fst.follow(fst.initial, vocabulary['abb-'])
        self.assertEqual(len(out), 1)

        _, out = fst.follow(state_1, vocabulary['---'])
        self.assertEqual(len(out), 1)

    def test_if_grammar(self):
        grammar = """
            ?start: sum
                | NAME "=" sum    

            ?sum: product
                | sum "+" product   
                | sum "-" product   
                | sum "---" product 

            ?product: atom
                | product "*" atom  
                | product "/" atom 

            ?atom: NUMBER           
                | "-" atom
                | IF       
                | NAME            
                | "(" sum ")"

            
            IF.0: "if"
            NAME.1: CNAME

            %import common.CNAME
            %import common.NUMBER
            %import common.WS_INLINE

            %ignore WS_INLINE
        """

        lexer_conf = Lark(grammar, parser='lalr').lexer_conf
        vocabulary = {'if ':1, 'iff':2, 'if':3, ' ':4}

        fst = PartialLexerFST(lexer_conf, vocabulary)

        state_1, out = fst.follow(fst.initial, vocabulary['iff'])
        self.assertEqual(len(out), 0)

        _, out = fst.follow(state_1, vocabulary[' '])
        self.assertEqual(len(out), 1)

        state_1, out = fst.follow(fst.initial, vocabulary['if'])
        self.assertEqual(len(out), 0)

        _, out = fst.follow(state_1, vocabulary['iff'])
        self.assertEqual(len(out), 0)

        _, out = fst.follow(state_1, vocabulary[' '])
        self.assertEqual(len(out), 1)

        print([(t.name, t.priority) for t in lexer_conf.terminals])
        self.assertEqual(out[0].name, 'IF')