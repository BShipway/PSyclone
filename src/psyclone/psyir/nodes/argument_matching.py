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

'''This module contains the rules deciding whether one actual argument of a
Call matches one dummy argument of a candidate Routine.

Matching is scored rather than answered yes or no. Fortran settles some
argument types only at the point of the call -- the kind of a literal is the
formal's kind, the shape a section is passed with is the formal's shape --
and the PSyIR does not always hold enough of the program to settle them for
itself. A rule that relaxed such a case silently would resolve a generic
interface by whichever specific it happened to try first; one that refused it
would decline calls that are perfectly legal Fortran. So a relaxed match is
reported as a match of score one, an exact match as score zero, and
:py:meth:`psyclone.psyir.nodes.Call.get_callee` takes the lowest-scoring
candidate and refuses a tie.

'''

from psyclone.errors import PSycloneError
from psyclone.psyir.nodes.datanode import DataNode
from psyclone.psyir.nodes.literal import Literal
from psyclone.psyir.symbols import (
    DataSymbol,
    DataTypeSymbol,
    ScalarType,
    UnsupportedFortranType,
    UnresolvedType,
)
from psyclone.psyir.symbols.datatypes import ArrayType

#: The score of a match that needed none of the relaxations below.
EXACT_MATCH = 0
#: The score of a match that needed one of them.
WEAK_MATCH = 1


class CallMatchingArgumentsNotFound(PSycloneError):
    '''Exception to signal that matching arguments have not been found
    for this routine
    '''
    def __init__(self, value):
        PSycloneError.__init__(self, value)
        self.value = "CallMatchingArgumentsNotFound: " + str(value)


def _type_symbols_match(type1, type2) -> bool:
    '''
    :param type1: the first type to compare.
    :type type1: :py:class:`psyclone.psyir.symbols.DataTypeSymbol` |
        :py:class:`psyclone.psyir.symbols.DataType`
    :param type2: the second type to compare.
    :type type2: :py:class:`psyclone.psyir.symbols.DataTypeSymbol` |
        :py:class:`psyclone.psyir.symbols.DataType`

    :returns: True if the two types correspond to DataTypeSymbols with
        the same name (case insensitive), False otherwise.

    '''
    return (isinstance(type1, DataTypeSymbol) and
            isinstance(type2, DataTypeSymbol) and
            (type1.name.lower() == type2.name.lower()))


def _match_arrays(actual: DataNode, actual_type: ArrayType,
                  dummy: DataSymbol, dummy_type: ArrayType,
                  interface_call: bool) -> None:
    '''Check an array actual argument against an array dummy argument.

    Arguments are only required to have the same rank if we are dealing with
    an interface call (polymorphism) or the dummy argument does not have an
    explicit shape (in which case Fortran permits implicit reshaping). The
    extents themselves are never compared: the shape an actual is passed with
    is the dummy's shape, and the two are written as different expressions in
    different scopes even when they are the same number.

    :param actual: one argument of the call.
    :param actual_type: the type of that argument.
    :param dummy: the corresponding argument of the candidate routine.
    :param dummy_type: the type of that argument, which for a dummy the PSyIR
        does not fully model is its partial datatype.
    :param interface_call: whether the call is made through a generic
        interface, which is resolved by rank as well as by type.

    :raises CallMatchingArgumentsNotFound: if the ranks differ where they
        have to match, or the intrinsic types differ.

    '''
    # Is the dummy argument an explicit-shape array?
    has_explicit_shape = all(
        isinstance(dim, ArrayType.ArrayBounds) and
        dim.lower is not ArrayType.Extent.ATTRIBUTE and
        dim.upper is not ArrayType.Extent.ATTRIBUTE
        for dim in dummy_type.shape)
    match_rank = interface_call or not has_explicit_shape
    if match_rank and len(actual_type.shape) != len(dummy_type.shape):
        raise CallMatchingArgumentsNotFound(
            f"Rank mismatch of call argument "
            f"'{actual.debug_string().strip()}' "
            f"(rank {len(actual_type.shape)}) and routine argument "
            f"'{dummy.name}' (rank {len(dummy_type.shape)})")
    # Arguments must have the same intrinsic type.
    if actual_type.intrinsic != dummy_type.intrinsic:
        if _type_symbols_match(actual_type.intrinsic, dummy_type.intrinsic):
            return
        raise CallMatchingArgumentsNotFound(
            f"Array argument type mismatch of call argument "
            f"'{actual.debug_string().strip()}' ({actual_type.intrinsic}) and "
            f"routine argument '{dummy.name}' ({dummy_type.intrinsic})")


def _match_partial(actual: DataNode, actual_type, dummy: DataSymbol,
                   dummy_type: UnsupportedFortranType,
                   interface_call: bool) -> None:
    '''Check an actual argument against a dummy the PSyIR does not fully
    model -- an 'optional' argument, say. Such a dummy carries at least a
    partial datatype, and that is what the actual is checked against.

    Where the partial datatype is an array, the array rule is applied to it
    (R3). Comparing the two types outright cannot succeed there: the actual
    is a section of one array and the dummy an explicit shape written in the
    callee's own scope, so the comparison is of two shape expressions that
    are never equal even when they describe the same extent.

    :param actual: one argument of the call.
    :param actual_type: the type of that argument.
    :type actual_type: :py:class:`psyclone.psyir.symbols.DataType`
    :param dummy: the corresponding argument of the candidate routine.
    :param dummy_type: the type of that argument.
    :param interface_call: whether the call is made through a generic
        interface, which is resolved by rank as well as by type.

    :raises CallMatchingArgumentsNotFound: if the two arguments do not match.

    '''
    partial_type = dummy_type.partial_datatype
    if isinstance(actual_type, ArrayType) and isinstance(partial_type,
                                                         ArrayType):
        _match_arrays(actual, actual_type, dummy, partial_type,
                      interface_call)
        return
    if actual_type != partial_type:
        raise CallMatchingArgumentsNotFound(
            f"Argument partial type mismatch of call argument "
            f"'{actual.debug_string().strip()}' ({actual_type}) and "
            f"routine argument '{dummy.name}' ({partial_type})")


def _is_kindless_literal(actual: DataNode, actual_type, dummy_type) -> bool:
    '''Say whether the actual is a literal whose kind the dummy supplies.

    A Fortran literal written without a kind takes the default kind of its
    intrinsic type, and the compiler has already accepted the call, so where
    the two agree on the intrinsic type the kinds agree too. The PSyIR
    records the literal's kind as ``UNDEFINED`` and cannot conclude that for
    itself, which is why a match found this way is a weak one.

    :param actual: one argument of the call.
    :param actual_type: the type of that argument.
    :type actual_type: :py:class:`psyclone.psyir.symbols.DataType`
    :param dummy_type: the type of the corresponding routine argument.
    :type dummy_type: :py:class:`psyclone.psyir.symbols.DataType`

    :returns: True if the two are scalars of the same intrinsic type and the
        actual is a literal that states no kind.

    '''
    return (isinstance(actual, Literal) and
            isinstance(actual_type, ScalarType) and
            isinstance(dummy_type, ScalarType) and
            actual_type.precision is ScalarType.Precision.UNDEFINED and
            actual_type.intrinsic == dummy_type.intrinsic)


def match_argument(actual: DataNode, dummy: DataSymbol, *,
                   interface_call: bool) -> int:
    '''Score one actual argument of a call against one dummy argument of a
    candidate routine.

    Three matches are relaxed, each of them a rule of Fortran the types alone
    do not carry:

    * an actual whose type is unresolved -- its symbol comes from a module
      PSyclone has not read -- could be of any type, so it matches any dummy
      weakly rather than refusing every candidate;
    * a literal that states no kind takes the dummy's kind, so it matches any
      kind of its own intrinsic type, weakly;
    * an array actual passed to a dummy the PSyIR does not fully model is
      compared with the dummy's partial datatype by the array rule, rather
      than by comparing two shape expressions written in different scopes.
      That one is exact: it relaxes nothing Fortran requires.

    :param actual: one argument of the call.
    :param dummy: the corresponding argument of the candidate routine.
    :param interface_call: whether the call is made through a generic
        interface, which is resolved by rank as well as by type.

    :returns: :py:data:`EXACT_MATCH` if the types match outright, or
        :py:data:`WEAK_MATCH` if the match needed one of the relaxations.

    :raises CallMatchingArgumentsNotFound: if the two arguments do not match.

    '''
    actual_type = actual.datatype
    dummy_type = dummy.datatype

    if isinstance(actual_type, UnresolvedType):
        return WEAK_MATCH

    if isinstance(actual_type, ArrayType) and isinstance(dummy_type,
                                                         ArrayType):
        _match_arrays(actual, actual_type, dummy, dummy_type, interface_call)
        return EXACT_MATCH

    if isinstance(dummy_type, UnsupportedFortranType):
        _match_partial(actual, actual_type, dummy, dummy_type, interface_call)
        return EXACT_MATCH

    if actual_type != dummy_type:
        if _type_symbols_match(actual_type, dummy_type):
            return EXACT_MATCH
        if _is_kindless_literal(actual, actual_type, dummy_type):
            return WEAK_MATCH
        raise CallMatchingArgumentsNotFound(
            f"Argument type mismatch of call argument "
            f"'{actual.debug_string().strip()}' ({actual_type}) and routine "
            f"argument '{dummy.name}' ({dummy_type})")

    return EXACT_MATCH


# For AutoAPI documentation generation
__all__ = ["CallMatchingArgumentsNotFound", "match_argument",
           "EXACT_MATCH", "WEAK_MATCH"]
