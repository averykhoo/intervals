# itf1788 vectors

the seven `.itl` files here are unmodified copies from [ITF1788](https://github.com/nehmeier/ITF1788),
commit `e0e0d7e7335e261e0dff547d15f2bfd3d1a612fa` (2015-02-16): `libieeep1788_tests_elem.itl` and
`libieeep1788_tests_set.itl` fetched 2026-09-24; `libieeep1788_tests_bool.itl`, `_num.itl`,
`_overlap.itl`, `_rec_bool.itl` and `atan2.itl` fetched 2026-09-25. each file's git blob hash equals
upstream's at that commit. ITF1788 is licensed under the Apache License 2.0: its `LICENSE` and
`NOTICE` are copied here unchanged, and each `.itl` file keeps its own copyright header.

`test_itf1788.py` runs every vector of the ops the package implements through the conformance
adapter described in its docstring (and in `v2-plan.md`, "ieee 1788"); `itl.py` is the parser. the
other ops in these files (`pow`, `less`, `interior`, `mid`, `wid`, ...) are counted and skipped. the
reverse-op files (`*_rev.itl`) and `libieeep1788_tests_cancel.itl` are not vendored: no op in them is
implemented.
