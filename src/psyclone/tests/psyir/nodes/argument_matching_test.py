# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

''' Performs py.test tests on the argument-matching rules used when
resolving the callee of a Call. '''

import pytest

from psyclone.psyir.nodes import Literal, Reference
from psyclone.psyir.nodes.argument_matching import (
    CallMatchingArgumentsNotFound, match_argument)
from psyclone.psyir.symbols import (
    ArrayType, DataSymbol, ScalarType, UnsupportedFortranType, UnresolvedType)


#: An integer literal written without a kind, as Fortran permits.
_KINDLESS_INT = ScalarType(ScalarType.Intrinsic.INTEGER,
                           ScalarType.Precision.UNDEFINED)
#: The same intrinsic type, declared with an explicit kind.
_KIND8_INT = ScalarType(ScalarType.Intrinsic.INTEGER, 8)


def test_match_argument_exact_match_scores_zero():
    '''An actual and a dummy of exactly the same type score zero.'''
    actual = Literal("1", _KINDLESS_INT)
    dummy = DataSymbol("a", _KINDLESS_INT)
    assert match_argument(actual, dummy, interface_call=False) == 0


def test_match_argument_intrinsic_mismatch_still_raises():
    '''A relaxation is not a licence to match across intrinsic types.'''
    actual = Literal("1", _KINDLESS_INT)
    dummy = DataSymbol("a", ScalarType(ScalarType.Intrinsic.REAL, 8))
    with pytest.raises(CallMatchingArgumentsNotFound) as err:
        match_argument(actual, dummy, interface_call=False)
    assert "Argument type mismatch of call argument '1'" in str(err.value)


def test_match_argument_unresolved_actual_is_a_weak_match():
    '''R1: an actual whose type is unresolved matches anything, weakly.'''
    actual = Reference(DataSymbol("chi", UnresolvedType()))
    dummy = DataSymbol("a", _KIND8_INT)
    assert match_argument(actual, dummy, interface_call=False) == 1


def test_match_argument_literal_without_a_kind_is_a_weak_match():
    '''R2: a literal states no kind, so any kind of that intrinsic type
    matches it, weakly.'''
    actual = Literal("1", _KINDLESS_INT)
    dummy = DataSymbol("a", _KIND8_INT)
    assert match_argument(actual, dummy, interface_call=False) == 1


def test_match_argument_literal_with_a_kind_is_not_relaxed():
    '''R2 applies to a literal that states no kind and to no other.'''
    actual = Literal("1", ScalarType(ScalarType.Intrinsic.INTEGER, 4))
    dummy = DataSymbol("a", _KIND8_INT)
    with pytest.raises(CallMatchingArgumentsNotFound) as err:
        match_argument(actual, dummy, interface_call=False)
    assert "Argument type mismatch of call argument '1_4'" in str(err.value)


def test_match_argument_a_reference_is_not_relaxed_by_r2():
    '''R2 is about a literal, which is the only thing whose kind Fortran
    takes from the formal.'''
    actual = Reference(DataSymbol("i", _KINDLESS_INT))
    dummy = DataSymbol("a", _KIND8_INT)
    with pytest.raises(CallMatchingArgumentsNotFound) as err:
        match_argument(actual, dummy, interface_call=False)
    assert "Argument type mismatch of call argument 'i'" in str(err.value)


def _explicit_shape_dummy(intrinsic=ScalarType.Intrinsic.REAL):
    '''
    :param intrinsic: the intrinsic type of the dummy's partial datatype.
    :type intrinsic: :py:class:`psyclone.psyir.symbols.ScalarType.Intrinsic`

    :returns: a dummy argument declared in a way the PSyIR does not model,
        but for which it has an explicit-shape partial datatype.
    :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
    '''
    partial = ArrayType(ScalarType(intrinsic, ScalarType.Precision.UNDEFINED),
                        [10])
    return DataSymbol("x", UnsupportedFortranType(
        "REAL, DIMENSION(nlayers), TARGET :: x", partial_datatype=partial))


def test_match_argument_section_against_an_explicit_shape_dummy():
    '''R3: a section of an array is matched against the dummy's partial
    datatype by the array rule, not by comparing two shape expressions.'''
    array = ArrayType(ScalarType(ScalarType.Intrinsic.REAL,
                                 ScalarType.Precision.UNDEFINED), [20])
    actual = Reference(DataSymbol("field", array))
    assert match_argument(actual, _explicit_shape_dummy(),
                          interface_call=False) == 0


def test_match_argument_section_of_the_wrong_intrinsic_raises():
    '''R3 relaxes the shape and nothing else.'''
    array = ArrayType(ScalarType(ScalarType.Intrinsic.REAL,
                                 ScalarType.Precision.UNDEFINED), [20])
    actual = Reference(DataSymbol("field", array))
    dummy = _explicit_shape_dummy(ScalarType.Intrinsic.INTEGER)
    with pytest.raises(CallMatchingArgumentsNotFound) as err:
        match_argument(actual, dummy, interface_call=False)
    assert "Array argument type mismatch of call argument 'field'" in str(
        err.value)


def test_match_argument_section_rank_is_checked_for_an_interface_call():
    '''A generic interface is resolved by rank as well as by type, so the
    relaxed comparison still checks it.'''
    array = ArrayType(ScalarType(ScalarType.Intrinsic.REAL,
                                 ScalarType.Precision.UNDEFINED), [20, 20])
    actual = Reference(DataSymbol("field", array))
    with pytest.raises(CallMatchingArgumentsNotFound) as err:
        match_argument(actual, _explicit_shape_dummy(), interface_call=True)
    assert "Rank mismatch of call argument 'field'" in str(err.value)


def test_match_argument_partial_type_mismatch_still_raises():
    '''A dummy the PSyIR does not model, whose partial datatype is not an
    array, is compared as it was before.'''
    dummy = DataSymbol("x", UnsupportedFortranType(
        "INTEGER, POINTER :: x", partial_datatype=_KIND8_INT))
    actual = Reference(DataSymbol("i", _KINDLESS_INT))
    with pytest.raises(CallMatchingArgumentsNotFound) as err:
        match_argument(actual, dummy, interface_call=False)
    assert "Argument partial type mismatch of call argument 'i'" in str(
        err.value)
