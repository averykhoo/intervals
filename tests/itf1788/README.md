# itf1788 vectors

the nineteen `.itl` files here are unmodified copies of every file under `itl/` in
[oheim/ITF1788](https://github.com/oheim/ITF1788), the maintained fork of
[nehmeier/ITF1788](https://github.com/nehmeier/ITF1788), at commit
`b6ee1e24d209c289f99a68ddc357839935799eae` (2018-09-22), fetched 2026-09-25. they replace the seven
files from nehmeier's `e0e0d7e` that were here before: the fork renames those
(`libieeep1788_tests_elem.itl` is `libieeep1788_elem.itl`, and so on) and corrects and extends them.
`LICENSE`, `NOTICE` and `COPYING.LESSER` are the fork's, from the repository root at the same
commit, also unmodified. `.gitattributes` marks all of them `-text`, so git never changes their line
endings: the working-tree files are the upstream bytes.

## licences

each `.itl` file keeps its own copyright and licence header. read from the headers (2026-09-25):

| files | licence |
|---|---|
| the eleven `libieeep1788_*.itl` | Apache License 2.0 (`LICENSE`, `NOTICE`) |
| `mpfi.itl`, `fi_lib.itl`, `c-xsc.itl` | GNU LGPL 2.1 or (at your option) any later version (`COPYING.LESSER`); vectors converted from those libraries' own test suites |
| `ieee1788-constructors.itl`, `ieee1788-exceptions.itl`, `atan2.itl`, `abs_rev.itl`, `pow_rev.itl` | all-permissive: "Copying and distribution of this file, with or without modification, are permitted in any medium without royalty provided the copyright notice and this notice are preserved. This file is offered as-is, without any warranty." |

these files are test data only: the wheel ships `intervals/` alone, so none of them is distributed
with the library.

## the hash check

each file's git blob hash equals the blob in the fork's tree at the pinned commit (checked
2026-09-25, all 22 files). to re-run it from the repository root:

```bash
python - <<'EOF'
import json, subprocess, urllib.request
url = 'https://api.github.com/repos/oheim/ITF1788/git/trees/b6ee1e24d209c289f99a68ddc357839935799eae?recursive=1'
for e in json.load(urllib.request.urlopen(url))['tree']:
    name = e['path'].removeprefix('itl/')
    if e['path'].startswith('itl/') or name in ('LICENSE', 'NOTICE', 'COPYING.LESSER'):
        ours = subprocess.run(['git', 'hash-object', f'tests/itf1788/{name}'], capture_output=True, text=True).stdout.strip()
        print('ok  ' if ours == e['sha'] else 'DIFF', name)
EOF
```

it prints `ok` for all 22 files (python 3.9 or later; the GitHub API needs no token for this).

## how they are used

`test_itf1788.py` runs every vector of the ops the package implements through the conformance
adapter described in its docstring (and in `v2-plan.md`, "ieee 1788"); `itl.py` is the parser, which
reads every statement of every file. the statements of ops not implemented yet (the reverse ops, the
text constructors, the decoration ops) are counted in `SKIPPED` and not run.
