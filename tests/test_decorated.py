"""
ieee 1788's decorated type (M13g): `intervals.decorated`, `DecoratedInterval`, `Decoration`, `set_dec`
and the decorated constructors

* the decorations against 1788's definitions, written out here: the empty set fits `trv` only, `com`
  needs a non-empty set with no point at and no piece reaching ±inf, `dac` and `def` any non-empty
  set. `DecoratedInterval(x)` (newDec) is the best that fits (maximality: every better one is
  refused), `DecoratedInterval(x, d)` constructs iff `d` fits, and `set_dec(x, d)` is `d` where it
  fits, else 1788's demotion (`trv` for the empty set, `dac` for `com` on an unbounded one): never a
  promotion, the best that fits at or below `d`
* the order `trv < def < dac < com`, total, so the weaker of two is `min`
* `ill`, NaI's decoration, and any other name raise `UndefinedOperationError` (no NaI, D16); a
  non-decoration or a non-`MultiInterval` is a `TypeError`
* the constructors are the bare ones plus a decoration: a text with no decoration gets newDec's, a
  fitting one is kept, an unfitting one raises; the numbers get newDec's. over bracket literals, the
  uncertain form and any text over the literal alphabet
* round trips: `str` of a bounded connected closed exact set reads back through
  `text_to_decorated_interval`; `repr` evaluates back, and pickle and copy keep it; the wrapper
  keeps the set's class (`OutwardMultiInterval`) and the set itself
* the itf1788 vectors of `d-textToInterval`, `d-numsToInterval`, `newDec`, `setDec`, `intervalPart`,
  `decorationPart` run in tests/itf1788; the ones that matter are examples here
"""
import copy
import math
import pickle
from fractions import Fraction

import pytest
from hypothesis import example
from hypothesis import given
from hypothesis import strategies as st

import intervals
from intervals import DecoratedInterval
from intervals import Decoration
from intervals import MultiInterval
from intervals import OutwardMultiInterval
from intervals import UndefinedOperationError
from intervals import nums_to_decorated_interval
from intervals import nums_to_interval
from intervals import set_dec
from intervals import text_to_decorated_interval
from intervals import text_to_interval
from tests.strategies import cut_tuples
from tests.strategies import exact_cut_tuples
from tests.test_literals import _ALPHABET
from tests.test_literals import bounds
from tests.test_literals import bracket_literals
from tests.test_literals import build
from tests.test_literals import uncertain

M = MultiInterval
INF = math.inf
TRV, DEF, DAC, COM = Decoration.TRV, Decoration.DEF, Decoration.DAC, Decoration.COM
WORST_TO_BEST = [TRV, DEF, DAC, COM]

# sets with open and closed ends, several pieces, ±inf as points and as unattained ends, and ends past
# the doubles (an exact 10**400 is bounded)
_huge = st.sampled_from([10 ** 400, -10 ** 400, Fraction(10 ** 400, 3)])
sets = st.one_of(cut_tuples(), cut_tuples(values=st.one_of(_huge, st.integers(-3, 3),
                                                           st.sampled_from([-INF, INF])))).map(M.from_cuts)
decorations = st.sampled_from(WORST_TO_BEST)


def bounded(x: MultiInterval) -> bool:
    """1788's bounded, on the set: no end of any piece is ±inf"""
    return all(-INF < cut.value < INF for cut in x.cuts)


def fits(x: MultiInterval, d: Decoration) -> bool:
    """1788's valid (interval, decoration) pairs, with no NaI: empty is trv only, com is bounded only"""
    if d is TRV:
        return True
    return bool(x) and (d is not COM or bounded(x))


# NEWDEC AND THE CONSTRUCTOR

@given(sets)
@example(M())  # libieeep1788_class.itl:264: newDec [empty] = [empty]_trv
@example(M.parse('(-inf, inf)'))  # :263: newDec [entire] = _dac
@example(M(-1.7976931348623157e308, 1.7976931348623157e308))  # :261: [-max, max] is com
@example(M(10 ** 400))  # bounded as a rational: com here, where 1788's binary64 hull is dac (:165)
@example(M.parse('[inf]'))  # an attained infinity is unbounded
@example(M.parse('{ [1, 2] , [3] }'))  # several pieces, bounded
def test_new_dec_is_the_best_decoration_that_fits(x):
    d = DecoratedInterval(x)
    assert d.interval is x
    assert fits(x, d.decoration)
    assert d.decoration is (TRV if not x else COM if bounded(x) else DAC)
    # maximality: every decoration fits up to newDec's and none past it, and the constructor agrees
    for e in WORST_TO_BEST:
        assert fits(x, e) == (e <= d.decoration)
        if fits(x, e):
            made = DecoratedInterval(x, e)
            assert made.interval is x and made.decoration is e
            assert DecoratedInterval(x, e.value) == made  # the name works as the member
        else:
            with pytest.raises(UndefinedOperationError):
                DecoratedInterval(x, e)


@pytest.mark.parametrize('x, d', [
    (M(), DEF), (M(), DAC), (M(), COM),  # the empty set is trv only
    (M(1, INF, end_closed=False), COM), (M.parse('[inf]'), COM), (M.parse('(-inf, inf)'), COM),
])
def test_the_constructor_refuses_what_does_not_fit(x, d):
    with pytest.raises(UndefinedOperationError):
        DecoratedInterval(x, d)
    with pytest.raises(UndefinedOperationError):
        DecoratedInterval(x, d.value)
    assert set_dec(x, d).decoration < d  # 1788's setDec demotes it instead


@given(sets, decorations, st.booleans())
@example(M(), DEF, False)  # libieeep1788_class.itl:283: setDec [empty] def = [empty]_trv
@example(M(), DAC, True)  # :284
@example(M(), COM, False)  # :285
@example(M(1, INF, end_closed=False), COM, True)  # :286: setDec [1.0,infinity] com = _dac
@example(M(-INF, 3, start_closed=False), COM, False)  # :287
@example(M.parse('(-inf, inf)'), COM, True)  # :288
@example(M.parse('(-inf, inf)'), DAC, False)  # :279
@example(M(), TRV, False)  # :280
@example(M(-1.7976931348623157e308), COM, True)  # :276
@example(M.parse('[1, inf]'), DEF, False)  # below newDec's: kept
def test_set_dec_is_1788s(x, d, by_name):
    result = set_dec(x, d.value if by_name else d)
    assert result.interval is x
    # 1788's definition, written out: d where it fits; else trv for the empty set, dac for com
    if fits(x, d):
        assert result.decoration is d
    elif not x:
        assert result.decoration is TRV
    else:
        assert d is COM and result.decoration is DAC
    # never promotes, and is the best that fits at or below d
    assert result.decoration <= d and fits(x, result.decoration)
    assert not any(result.decoration < e <= d and fits(x, e) for e in WORST_TO_BEST)
    # the laws: newDec's own decoration is kept, set_dec is min(d, newDec's), and it is idempotent
    best = DecoratedInterval(x).decoration
    assert set_dec(x, best) == DecoratedInterval(x)
    assert result.decoration is min(d, best)
    assert set_dec(x, result.decoration) == result
    if fits(x, d):
        assert result == DecoratedInterval(x, d)


@pytest.mark.parametrize('name', ['ill', 'ILL', 'nai', 'Com', 'COM', 'fooo', 'da', ''])
def test_ill_and_other_names_raise(name):
    # libieeep1788_class.itl:289-291: setDec x ill = [nai] signal UndefinedOperation
    for x in (M(), M(-1, 3), M(-INF, 3, start_closed=False)):
        with pytest.raises(UndefinedOperationError):
            set_dec(x, name)
        with pytest.raises(UndefinedOperationError):
            DecoratedInterval(x, name)


@pytest.mark.parametrize('decoration', [None, 3, b'com', ('com',), Decoration])
def test_a_non_decoration_is_a_type_error(decoration):
    with pytest.raises(TypeError):
        set_dec(M(1, 2), decoration)
    if decoration is not None:  # None is newDec's
        with pytest.raises(TypeError):
            DecoratedInterval(M(1, 2), decoration)


@pytest.mark.parametrize('interval', [1, 1.5, (1, 2), '[1, 2]', None, DecoratedInterval(M(1, 2))])
def test_a_non_multi_interval_is_a_type_error(interval):
    with pytest.raises(TypeError):
        DecoratedInterval(interval)
    with pytest.raises(TypeError):
        set_dec(interval, COM)


# THE ORDER

@given(decorations, decorations)
def test_the_order_is_trv_def_dac_com(a, b):
    i, j = WORST_TO_BEST.index(a), WORST_TO_BEST.index(b)
    assert (a < b, a <= b, a > b, a >= b, a == b) == (i < j, i <= j, i > j, i >= j, i == j)
    assert min(a, b) is WORST_TO_BEST[min(i, j)]


def test_the_order_is_only_among_decorations():
    assert sorted(Decoration) == WORST_TO_BEST and len(Decoration) == 4  # no ill
    for op in ('__lt__', '__le__', '__gt__', '__ge__'):
        assert getattr(COM, op)('com') is NotImplemented
    with pytest.raises(TypeError):
        _ = COM < 3
    assert COM != 'com'


# THE CONSTRUCTORS

@given(bracket_literals(), st.sampled_from([None, 'com', 'dac', 'def', 'trv', 'COM', 'Trv', 'ill', 'fooo']),
       st.sampled_from(['', ' ', '\t']))
@example((['empty'], (None, None)), 'trv', ' ')  # libieeep1788_class.itl:143
@example((['empty'], (None, None)), 'ill', ' ')  # :208
@example(([], (None, None)), None, ' ')  # :144, constructors.itl:57: [ ] = [empty]_trv
@example(([], (None, None)), 'com', ' ')  # :209
@example((['', ',', ''], (-INF, INF)), None, '')  # :147: [,] = [entire]_dac
@example((['', ',', ''], (-INF, INF)), 'trv', '')  # :148
@example((['', ',', ''], (-INF, INF)), 'com', '')  # :210
@example((['entire'], (-INF, INF)), 'com', ' ')  # :211
@example((['-inf', ',', 'INF'], (-INF, INF)), 'def', ' ')  # :152
@example((['-1.0', ',', ''], (-1, INF)), 'com', '')  # :216
@example((['-1.0', ',', '1.0'], (-1, 1)), 'COM', '  ')  # :155, constructors.itl:25
@example((['-1.0', ',', '1.0'], (-1, 1)), 'fooo', ' ')  # :214
@example((['1.0E+400'], (10 ** 400, 10 ** 400)), 'com', ' ')  # :165: com fits the exact value
def test_text_is_the_bare_constructor_and_a_decoration(case, decoration, space):
    parts, (lo, hi) = case
    plain = build(parts, space)
    text = plain if decoration is None else plain + '_' + decoration
    bare = text_to_interval(plain)
    name = None if decoration is None else decoration.lower()
    if name is None:
        assert text_to_decorated_interval(text) == DecoratedInterval(bare)
    elif name in ('com', 'dac', 'def', 'trv') and fits(bare, Decoration(name)):
        result = text_to_decorated_interval(text)
        assert result.interval == bare and result.decoration is Decoration(name)
    else:
        with pytest.raises(UndefinedOperationError):
            text_to_decorated_interval(text)


@given(uncertain(), st.sampled_from([None, 'com', 'dac', 'def', 'trv']))
@example(('3.56?1', Fraction(355, 100), Fraction(357, 100), Fraction(356, 100), Fraction(1, 100)), 'def')
@example(('0.0??', -INF, INF, 0, Fraction(1, 10)), 'com')  # libieeep1788_class.itl:223
@example(('0.0??d', -INF, 0, 0, Fraction(1, 10)), None)  # :188: = [-infinity,0.0]_dac
@example(('0.0??d', -INF, 0, 0, Fraction(1, 10)), 'com')  # :225
@example(('10?3e380', 7 * 10 ** 380, 13 * 10 ** 380, 10 ** 381, 10 ** 380), 'com')  # :204: com here
@example(('-10??u', -10, INF, -10, 1), None)  # constructors.itl:71: = [-10.0, infinity]_dac
def test_uncertain_form_with_a_decoration(case, decoration):
    text, lo, hi = case[:3]
    bare = text_to_interval(text)
    if decoration is None:
        assert text_to_decorated_interval(text) == DecoratedInterval(bare)
    elif fits(bare, Decoration(decoration)):
        assert text_to_decorated_interval(text + '_' + decoration) == DecoratedInterval(bare, decoration)
    else:
        with pytest.raises(UndefinedOperationError):
            text_to_decorated_interval(text + '_' + decoration)


@given(st.text(_ALPHABET.replace('_', ''), max_size=24))
@example('[nai]')  # constructors.itl:73: no NaI
@example('[ Nai  ]')  # libieeep1788_class.itl:198
@example('[1.0,2.0')  # :227
@example('[1.0000000000000002,1.0000000000000001]')  # :229: invalid as rationals
def test_any_undecorated_text_is_the_bare_one_with_new_dec(text):
    try:
        bare = text_to_interval(text)
    except UndefinedOperationError:
        with pytest.raises(UndefinedOperationError):
            text_to_decorated_interval(text)
    else:
        assert text_to_decorated_interval(text) == DecoratedInterval(bare)


@given(bounds, bounds)
@example(-1.0, 1.0)  # libieeep1788_class.itl:37: = [-1.0,1.0]_com
@example(-INF, 1.0)  # :38: = _dac
@example(-INF, INF)  # :40
@example(math.nan, math.nan)  # :42: [nai] signal UndefinedOperation
@example(1.0, -1.0)  # :43
@example(-INF, -INF)  # :44
@example(2, 1)  # constructors.itl:52
def test_nums_is_the_bare_constructor_with_new_dec(lo, hi):
    try:
        bare = nums_to_interval(lo, hi)
    except UndefinedOperationError:
        with pytest.raises(UndefinedOperationError):
            nums_to_decorated_interval(lo, hi)
    else:
        assert nums_to_decorated_interval(lo, hi) == DecoratedInterval(bare)


# ROUND TRIPS, IDENTITY, CLASS

@given(st.one_of(st.integers(-10 ** 30, 10 ** 30), st.fractions(max_denominator=10 ** 6), _huge),
       st.one_of(st.integers(0, 10 ** 30), st.fractions(min_value=0, max_denominator=10 ** 6)), decorations)
@example(-2, 4, COM)  # libieeep1788_class.itl:167
@example(Fraction(-1, 10), Fraction(1, 5), COM)  # :168, exactly
@example(10 ** 400, 0, COM)
def test_str_reads_back_as_1788_text(lo, width, d):
    x = M(lo, lo + width)
    assert text_to_decorated_interval(str(DecoratedInterval(x, d))) == DecoratedInterval(x, d)


@given(sets, decorations)
def test_repr_pickle_and_copy_give_it_back(x, d):
    d = set_dec(x, d)
    namespace = {'DecoratedInterval': DecoratedInterval, 'Decoration': Decoration, 'MultiInterval': MultiInterval}
    assert eval(repr(d), namespace) == d
    assert pickle.loads(pickle.dumps(d)) == d and copy.deepcopy(d) == d


@given(sets, sets, decorations, decorations)
def test_equality_is_both_parts(x, y, d, e):
    a, b = set_dec(x, d), set_dec(y, e)
    assert (a == b) == (x == y and a.decoration is b.decoration)
    assert (a != b) == (not a == b)
    if a == b:
        assert hash(a) == hash(b)
    assert a != x and not a == x  # never equal to the bare set


@given(cut_tuples(), decorations)
def test_the_set_keeps_its_class(cuts, d):
    x = OutwardMultiInterval.from_cuts(cuts)
    assert type(DecoratedInterval(x).interval) is OutwardMultiInterval
    assert set_dec(x, d).interval is x


@given(exact_cut_tuples)
def test_immutable(cuts):
    d = DecoratedInterval(M.from_cuts(cuts))
    with pytest.raises(AttributeError):
        d.decoration = TRV
    with pytest.raises(AttributeError):
        d._interval = M()
    with pytest.raises(AttributeError):
        del d._decoration


def test_str():
    assert str(DecoratedInterval(M(1, 2))) == '[1, 2]_com'
    assert str(set_dec(M.parse('(-inf, 0] | [1]'), DEF)) == '{ (-inf, 0] , [1] }_def'
    assert str(DecoratedInterval(M())) == '{}_trv'


def test_exported():
    for name in ('DecoratedInterval', 'Decoration', 'set_dec', 'text_to_decorated_interval',
                 'nums_to_decorated_interval'):
        assert name in intervals.__all__
