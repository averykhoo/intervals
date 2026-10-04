# verify construct-4 (docs lens): empty-by-flags `[a, a)`, `(a, a)`, `(a, a]` and their strings

claim: UNDOCUMENTED_DIFFERENCE (v1 ValueError, v2 empty set; only kernel.py::piece docstring records it).

## verdict: REFUTED -> DIFFERS_DOCUMENTED

the behaviour is named exactly, as a design decision, in the CURRENT design section of v2-plan.md:

* v2-plan.md:86-87, section "## current design (2026-09-23 ...)" > "### representation: cuts" (line 69):
  "empty iff `start >= end`: `[1,1)` and `(1,1)` normalize to empty. reversed *values* (`[2,1]`) are a ValueError"
* v2-plan.md:2560, "### v2 consolidated decisions (2026-08-16)" > "#### representation: cuts" (marked
  "partly superseded 2026-09-22: cuts ... survive"):
  "empty iff start_cut >= end_cut, so `[1,1)` and `(1,1)` normalize to empty; `[2,1]` is still ValueError"
* also: intervals/kernel.py:40 (piece docstring) "`[1, 1)` is an (empty) pair"; references/gemini-conversation-recap.md:235-238
  "Input `(1, 1)` ... Normalize to Empty Set. Do not raise Error." (design derivation chat, adjacent background).
* not a Q21 item (Q21 lists M8 time-layer choices); this is core representation, settled.

the claim's own examples are verbatim the doc's examples. it is a deliberate departure from v1's ValueError.

## re-run (2026-10-04)
probe: .scratch/v1-parity/verify/construct-4-docs-probe.py, construct-4-docs-probe-str.py
cmd: timeout 120 C:/Users/user/anaconda3/envs/intervals/python.exe <file>
* (1,1,start_closed=False): v1 ValueError "start (1, 1) is after end (1, 0)"; v2 {} is_empty=True
* (1,1,end_closed=False): v1 ValueError; v2 {}
* (1,1, both open): v1 ValueError; v2 {}
* M.parse: '(1, 1)' {} ; '[1, 1)' {} ; '(1, 1]' {} ; '[1, 1]' [1] ; '[2, 1]' ValueError (reversed values still raise, as documented)
* failing-able check: asserts parse('[1, 1]') non-empty and parse('[1, 1)') empty both pass.
