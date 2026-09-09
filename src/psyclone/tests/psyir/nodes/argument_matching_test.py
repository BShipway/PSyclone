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

from psyclone.psyir.nodes import (
    BinaryOperation, Container, Literal, Reference, Routine)
from psyclone.psyir.nodes.argument_matching import (
    CallMatchingArgumentsNotFound, match_argument)
from psyclone.psyir.symbols import (
    ArrayType, ContainerSymbol, DataSymbol, DataTypeSymbol, ImportInterface,
    ScalarType, UnsupportedFortranType, UnresolvedType)


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


def _kind(name, value=None):
    '''
    :param str name: the name of the named constant giving a kind.
    :param value: what the constant is defined as, if anything.
    :type value: Optional[:py:class:`psyclone.psyir.nodes.DataNode`]

    :returns: a reference to a named constant, as the PSyIR records the kind
        of a declaration written with one.
    :rtype: :py:class:`psyclone.psyir.nodes.Reference`
    '''
    symbol = DataSymbol(name, ScalarType(ScalarType.Intrinsic.INTEGER, 4),
                        is_constant=value is not None, initial_value=value)
    return Reference(symbol)


def _real_array(precision, rank=1):
    '''
    :param precision: the kind of the array's elements.
    :type precision: int | :py:class:`psyclone.psyir.nodes.DataNode` |
        :py:class:`psyclone.psyir.symbols.ScalarType.Precision`
    :param int rank: how many dimensions the array has.

    :returns: the type of a real array of that kind.
    :rtype: :py:class:`psyclone.psyir.symbols.ArrayType`
    '''
    return ArrayType(ScalarType(ScalarType.Intrinsic.REAL, precision),
                     [10] * rank)


def _integer_literal(value):
    '''
    :param int value: the value of the literal.

    :returns: an integer literal, as a named constant's definition holds it.
    :rtype: :py:class:`psyclone.psyir.nodes.Literal`
    '''
    return Literal(str(value), ScalarType(ScalarType.Intrinsic.INTEGER, 4))


def test_match_argument_arrays_of_the_same_kind_score_zero():
    '''R4: two arrays whose kinds resolve to the same value are an exact
    match, even though the two declarations name different constants.'''
    actual = Reference(DataSymbol(
        "chi", _real_array(_kind("r_def", _integer_literal(8)))))
    dummy = DataSymbol(
        "x", _real_array(_kind("r_double", _integer_literal(8))))
    assert match_argument(actual, dummy, interface_call=True) == 0


def test_match_argument_arrays_of_different_kinds_do_not_match():
    '''R4: two arrays whose kinds resolve to different values are not a
    match at all, which is what tells a generic interface's specifics
    apart.'''
    actual = Reference(DataSymbol(
        "chi", _real_array(_kind("r_def", _integer_literal(8)))))
    dummy = DataSymbol(
        "x", _real_array(_kind("r_single", _integer_literal(4))))
    with pytest.raises(CallMatchingArgumentsNotFound) as err:
        match_argument(actual, dummy, interface_call=True)
    assert ("Array kind mismatch of call argument 'chi' (kind 8) and routine "
            "argument 'x' (kind 4)" in str(err.value))


def test_match_argument_arrays_of_one_named_kind_score_zero():
    '''R4: a kind PSyclone cannot resolve still matches itself exactly, so
    the ordinary case of one kind used throughout costs nothing.'''
    actual = Reference(DataSymbol("chi", _real_array(_kind("r_def"))))
    dummy = DataSymbol("x", _real_array(_kind("r_def")))
    assert match_argument(actual, dummy, interface_call=True) == 0


def test_match_argument_arrays_of_unresolvable_kinds_are_a_weak_match():
    '''R4: two kinds named by constants PSyclone cannot resolve may or may
    not be the same value, so they match weakly rather than not at all.'''
    actual = Reference(DataSymbol("chi", _real_array(_kind("r_def"))))
    dummy = DataSymbol("x", _real_array(_kind("r_single")))
    assert match_argument(actual, dummy, interface_call=True) == 1


def test_match_argument_array_kind_chain_is_followed():
    '''R4: a named constant defined as another named constant is followed to
    the end of the chain, which is the shape LFRic's constants_mod has.'''
    real64 = _kind("real64")
    real32 = _kind("real32")
    actual = Reference(DataSymbol("chi", _real_array(_kind("r_def", real64))))
    dummy = DataSymbol("x", _real_array(_kind("r_single", real32)))
    with pytest.raises(CallMatchingArgumentsNotFound) as err:
        match_argument(actual, dummy, interface_call=True)
    assert ("(kind real64) and routine argument 'x' (kind real32)"
            in str(err.value))


def test_match_argument_array_kind_chain_that_closes_matches():
    '''R4: two chains that end at the same name are the same kind.'''
    actual = Reference(DataSymbol(
        "chi", _real_array(_kind("r_def", _kind("real64")))))
    dummy = DataSymbol(
        "x", _real_array(_kind("r_double", _kind("real64"))))
    assert match_argument(actual, dummy, interface_call=True) == 0


def test_match_argument_array_kind_number_against_a_name_is_weak():
    '''R4: an explicit number and a name PSyclone cannot resolve are not
    comparable, so the match is weak rather than refused.'''
    actual = Reference(DataSymbol("chi", _real_array(8)))
    dummy = DataSymbol("x", _real_array(_kind("r_def")))
    assert match_argument(actual, dummy, interface_call=True) == 1


def test_match_argument_array_kind_numbers_are_compared():
    '''R4: two explicit numbers are compared as they stand.'''
    actual = Reference(DataSymbol("chi", _real_array(8)))
    dummy = DataSymbol("x", _real_array(4))
    with pytest.raises(CallMatchingArgumentsNotFound) as err:
        match_argument(actual, dummy, interface_call=True)
    assert "(kind 8) and routine argument 'x' (kind 4)" in str(err.value)


def test_match_argument_array_of_default_kind_is_a_weak_match():
    '''R4: a declaration that states no kind takes the default one, which
    the PSyIR records as UNDEFINED rather than as a value, so it cannot be
    compared with a kind that is named.'''
    actual = Reference(DataSymbol(
        "chi", _real_array(ScalarType.Precision.UNDEFINED)))
    dummy = DataSymbol("x", _real_array(_kind("r_def", _integer_literal(8))))
    assert match_argument(actual, dummy, interface_call=True) == 1


def test_match_argument_arrays_of_default_kind_score_zero():
    '''R4: two declarations that both state no kind agree.'''
    actual = Reference(DataSymbol(
        "chi", _real_array(ScalarType.Precision.UNDEFINED)))
    dummy = DataSymbol(
        "x", _real_array(ScalarType.Precision.UNDEFINED))
    assert match_argument(actual, dummy, interface_call=True) == 0


def test_match_argument_array_kind_that_is_an_expression_is_weak():
    '''R4: a kind written as an expression is not reduced to a value, so it
    matches weakly.'''
    kind = BinaryOperation.create(
        BinaryOperation.Operator.MUL, _integer_literal(2),
        _integer_literal(4))
    actual = Reference(DataSymbol("chi", _real_array(kind)))
    dummy = DataSymbol("x", _real_array(_kind("r_def", _integer_literal(8))))
    assert match_argument(actual, dummy, interface_call=True) == 1


def test_match_argument_array_kind_that_is_not_a_number_is_weak():
    '''R4: a named constant whose value is not a whole number is not a kind
    PSyclone can compare.'''
    real_value = Literal("1.0", ScalarType(ScalarType.Intrinsic.REAL, 4))
    actual = Reference(DataSymbol("chi", _real_array(_kind("w", real_value))))
    dummy = DataSymbol("x", _real_array(_kind("r_def", _integer_literal(8))))
    assert match_argument(actual, dummy, interface_call=True) == 1


def test_match_argument_array_kind_chain_is_bounded():
    '''R4: a chain of named constants that never ends -- which only a
    malformed tree can be -- is abandoned rather than followed for ever.'''
    symbol = DataSymbol("wp", ScalarType(ScalarType.Intrinsic.INTEGER, 4))
    symbol._initial_value = Reference(symbol)   # pylint: disable=W0212
    actual = Reference(DataSymbol("chi", _real_array(Reference(symbol))))
    dummy = DataSymbol("x", _real_array(_kind("r_def", _integer_literal(8))))
    assert match_argument(actual, dummy, interface_call=True) == 1


def test_match_argument_array_of_a_derived_type_is_unchanged():
    '''R4 is about intrinsic kinds; an array of a derived type carries no
    precision and is matched as it was before.'''
    array = ArrayType(DataTypeSymbol("field_type", UnresolvedType()), [10])
    actual = Reference(DataSymbol("chi", array))
    dummy = DataSymbol("x", ArrayType(
        DataTypeSymbol("field_type", UnresolvedType()), [10]))
    assert match_argument(actual, dummy, interface_call=True) == 0


def test_match_argument_array_kind_reached_through_an_import(fortran_reader):
    '''R4: a kind named by a constant of another module is resolved by
    reading that module, which is how LFRic's constants_mod is reached.'''
    psyir = fortran_reader.psyir_from_source('''
module constants_mod
  implicit none
  integer, parameter :: r_def = 8
  integer, parameter :: r_single = 4
end module constants_mod
module some_mod
  use constants_mod, only: r_def
  implicit none
contains
  subroutine main(chi)
    real(kind=r_def) :: chi(10)
    chi(1) = 0.0_r_def
  end subroutine main
end module some_mod
''')
    actual = psyir.walk(Routine)[0].symbol_table.lookup("chi")
    other = ContainerSymbol("constants_mod")
    r_single = DataSymbol("r_single",
                          ScalarType(ScalarType.Intrinsic.INTEGER, 4),
                          interface=ImportInterface(other))
    dummy = DataSymbol("x", _real_array(Reference(r_single)))
    reference = Reference(actual)
    psyir.walk(Routine)[0].children[0].lhs.replace_with(reference)
    with pytest.raises(CallMatchingArgumentsNotFound) as err:
        match_argument(reference, dummy, interface_call=True)
    assert "(kind 8) and routine argument 'x' (kind 4)" in str(err.value)


def test_match_argument_array_kind_of_an_unreadable_module_is_weak():
    '''R4: a kind imported from a module PSyclone cannot read stays a name,
    and a name it cannot resolve is a weak match.'''
    absent = ContainerSymbol("no_such_mod")
    imported = DataSymbol("r_def", ScalarType(ScalarType.Intrinsic.INTEGER,
                                              4),
                          interface=ImportInterface(absent))
    actual = Reference(DataSymbol("chi", _real_array(Reference(imported))))
    dummy = DataSymbol("x", _real_array(_kind("r_single",
                                              _integer_literal(4))))
    assert match_argument(actual, dummy, interface_call=True) == 1


def test_match_argument_arrays_of_one_derived_type_are_unchanged():
    '''R4: two arrays of the very same derived type reach the kind
    comparison with no kind on either side, and are left an exact match.'''
    field_type = DataTypeSymbol("field_type", UnresolvedType())
    actual = Reference(DataSymbol("chi", ArrayType(field_type, [10])))
    dummy = DataSymbol("x", ArrayType(field_type, [10]))
    assert match_argument(actual, dummy, interface_call=True) == 0


def test_match_argument_array_kind_of_a_missing_module_is_weak(monkeypatch):
    '''R4: where the module giving a kind is not found -- rather than found
    and unreadable -- the kind stays a name and the match is weak.'''
    monkeypatch.setattr(ContainerSymbol, "find_container_psyir",
                        lambda self, local_node=None: None)
    other = ContainerSymbol("constants_mod")
    imported = DataSymbol("r_def", ScalarType(ScalarType.Intrinsic.INTEGER,
                                              4),
                          interface=ImportInterface(other))
    actual = Reference(DataSymbol("chi", _real_array(Reference(imported))))
    dummy = DataSymbol("x", _real_array(_kind("r_single",
                                              _integer_literal(4))))
    assert match_argument(actual, dummy, interface_call=True) == 1


def test_match_argument_array_kind_absent_from_its_module_is_weak(
        monkeypatch):
    '''R4: a module that is read but does not define the constant the
    declaration names leaves the kind a name, and so a weak match.'''
    empty = Container("constants_mod")
    monkeypatch.setattr(ContainerSymbol, "find_container_psyir",
                        lambda self, local_node=None: empty)
    other = ContainerSymbol("constants_mod")
    imported = DataSymbol("r_def", ScalarType(ScalarType.Intrinsic.INTEGER,
                                              4),
                          interface=ImportInterface(other))
    actual = Reference(DataSymbol("chi", _real_array(Reference(imported))))
    dummy = DataSymbol("x", _real_array(_kind("r_single",
                                              _integer_literal(4))))
    assert match_argument(actual, dummy, interface_call=True) == 1
