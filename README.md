# Alignment

This repository contains implementations for constrained-aligned decoding.

## GreatGramma

This repository also contains the implementation of `GreatGramma`, a tool implementing algorithm of the paper 'Flexible and Efficient Grammar-Constrained Decoding' `GreatGrammar` can be installed by running `pip install -e .`.
`GreatGramma` requires specific version of the `transformers` package. Run `pip install -r requirements.txt` for setup.

The CFG monitor is implemented in `alignment/monitor/grammar`. 
* `cfg_monitor.py` is frontend of the monitor
* `partial_lexer.pyx` is lexer preprocessing module
* `partial_parser.pyx` handles parser preprocessing and runtime parser check

`partial_lexer.pyx` and `partial_parser.pyx` need to compiled by running `cythonize -i alignment/monitor/grammar/partial_lexer.pyx` and `cythonize -i alignment/monitor/grammar/partial_parser.pyx`

### Known Issues

This implementation of `GreatGramma` is prototype and has memory leak issue. It is recommended to disable GC when grammar is huge.