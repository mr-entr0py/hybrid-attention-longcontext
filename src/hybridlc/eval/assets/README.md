# Vendored evaluation assets

`scowl_american_lower.txt`: lowercase a-z words (length >= 3) of SCOWL's `american-english`
(Debian package `wamerican` 2020.12.07-2, source file SHA-256
9f513f1ceadb6a01c5485b7dbdfd5118dc66cd70b59cae2851292112d4066a32), produced with

    LC_ALL=C grep -E '^[a-z]{3,}$' /usr/share/dict/american-english | LC_ALL=C sort -u

SHA-256 of the result: 37edcc1d0ae721dc10919159618edbd8ff5cae6f0149065bb8b6310a579f6932 (63,737 words).
Used only to keep tokenizer-vocabulary background words for A-freq / A-count to real words
(see `hybridlc/eval/words.py`).

SCOWL licence: "Copyright 2000-2011 by Kevin Atkinson. Permission to use, copy, modify,
distribute and sell these word lists, the associated scripts, the output created from the
scripts, and its documentation for any purpose is hereby granted without fee, provided that
the above copyright notice appears in all copies and that both that copyright notice and this
permission notice appear in supporting documentation. Kevin Atkinson makes no representations
about the suitability of this array for any purpose. It is provided "as is" without express or
implied warranty."
