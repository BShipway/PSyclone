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

from typing import Optional, Tuple, Union

from psyclone.errors import PSycloneError
from psyclone.psyir.nodes.datanode import DataNode
from psyclone.psyir.nodes.literal import Literal
from psyclone.psyir.nodes.reference import Reference
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

#: How far a chain of named constants defining a kind is followed before it
#: is given up on. A kind is written as a name defined as a name -- LFRic's
#: ``r_def`` is ``real64`` is an ``iso_fortran_env`` constant -- and the
#: chain is short in every real declaration; the bound is what stops a
#: malformed tree, in which a constant is defined as itself, being followed
#: for ever.
_KIND_CHAIN_LIMIT = 8


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
                  interface_call: bool) -> int:
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

    :returns: :py:data:`EXACT_MATCH` where the two kinds are the same or
        neither can be compared with the other, or :py:data:`WEAK_MATCH`
        where the kinds might or might not agree; see
        :py:func:`_match_kinds`.

    :raises CallMatchingArgumentsNotFound: if the ranks differ where they
        have to match, if the intrinsic types differ, or if the two kinds
        are known and different.

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
            return EXACT_MATCH
        raise CallMatchingArgumentsNotFound(
            f"Array argument type mismatch of call argument "
            f"'{actual.debug_string().strip()}' ({actual_type.intrinsic}) and "
            f"routine argument '{dummy.name}' ({dummy_type.intrinsic})")
    return _match_kinds(actual, actual_type.precision,
                        dummy, dummy_type.precision)


def _match_partial(actual: DataNode, actual_type, dummy: DataSymbol,
                   dummy_type: UnsupportedFortranType,
                   interface_call: bool) -> int:
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

    :returns: :py:data:`EXACT_MATCH`, or :py:data:`WEAK_MATCH` where the
        array rule was applied and the two kinds might or might not agree.

    :raises CallMatchingArgumentsNotFound: if the two arguments do not match.

    '''
    partial_type = dummy_type.partial_datatype
    if isinstance(actual_type, ArrayType) and isinstance(partial_type,
                                                         ArrayType):
        return _match_arrays(actual, actual_type, dummy, partial_type,
                             interface_call)
    if actual_type != partial_type:
        raise CallMatchingArgumentsNotFound(
            f"Argument partial type mismatch of call argument "
            f"'{actual.debug_string().strip()}' ({actual_type}) and "
            f"routine argument '{dummy.name}' ({partial_type})")
    return EXACT_MATCH


def _imported_initial_value(symbol: DataSymbol,
                            local_node: DataNode) -> Optional[DataNode]:
    '''Read what another module defines a named constant as.

    A kind is normally written with a constant of a module the declaration
    ``use``s, and the symbol the call site holds for it carries no value: it
    is a record that the name is imported, not a copy of the definition. The
    definition is in the other module's own symbol table, and is read from
    there rather than merged into this one, so that scoring a candidate
    leaves no mark on either tree.

    :param symbol: the named constant a kind is written with.
    :param local_node: a node of the tree the call is in, whose file is
        searched for the module before the search path is.

    :returns: what the module defines the constant as, or None if the module
        cannot be read or does not define it.

    '''
    if not symbol.is_import:
        return None
    try:
        container = symbol.interface.container_symbol.find_container_psyir(
            local_node=local_node)
    # pylint: disable-next=broad-except
    except Exception:
        # find_container_psyir reads a file through the ModuleManager and the
        # frontend, either of which may raise for a module that is absent,
        # unreadable or not Fortran. None of that is an error here: it means
        # the kind cannot be resolved, which the caller answers weakly.
        return None
    if container is None:
        return None
    try:
        return container.symbol_table.lookup(symbol.name).initial_value
    except (KeyError, AttributeError):
        return None


def _reduced_kind(precision, local_node: DataNode) -> Tuple[
        Optional[Union[int, str]], bool]:
    '''Reduce the kind of a declaration to something two kinds can be
    compared by.

    A kind reaches the PSyIR as a number, as one of
    :py:class:`psyclone.psyir.symbols.ScalarType.Precision`'s members where
    the declaration named none, or as a reference to a named constant. A
    named constant is followed to what it is defined as, and that to what
    *it* is defined as, until a number or a name with no definition to follow
    is reached: LFRic's ``r_def`` is ``real64``, which ``iso_fortran_env``
    defines and PSyclone cannot read.

    Whether the chain was followed at all is returned alongside the value it
    ended at, because the two answer different questions. Two names that were
    never resolved are two names, which may or may not be one value; two
    names each of which was followed to its definition are that definition,
    and where those differ the kinds differ.

    :param precision: the kind of a declaration.
    :type precision: Optional[int |
        :py:class:`psyclone.psyir.symbols.ScalarType.Precision` |
        :py:class:`psyclone.psyir.nodes.DataNode`]
    :param local_node: a node of the tree the call is in, used to find a
        module a kind is imported from.

    :returns: what the kind reduces to -- a number of bytes, the name it
        ends at, one of ``ScalarType.Precision``'s members, or None where it
        reduces to nothing comparable -- and whether reducing it read a
        definition.

    '''
    if isinstance(precision, ScalarType.Precision):
        # A declaration that names no kind takes a default this does not
        # know, so it is never evidence that two kinds differ.
        return (precision, False)
    if isinstance(precision, int):
        return (precision, True)
    node = precision
    resolved = False
    for _ in range(_KIND_CHAIN_LIMIT):
        if isinstance(node, Literal):
            try:
                return (int(node.value), True)
            except ValueError:
                return (None, False)
        if not isinstance(node, Reference):
            # An expression, which is not reduced: a kind is written as one
            # rarely enough that evaluating it here would be more machinery
            # than the answer is worth.
            return (None, False)
        symbol = node.symbol
        value = getattr(symbol, "initial_value", None)
        if value is None:
            value = _imported_initial_value(symbol, local_node)
        if value is None:
            return (symbol.name.lower(), resolved)
        resolved = True
        node = value
    return (None, False)


def _match_kinds(actual: DataNode, actual_precision,
                 dummy: DataSymbol, dummy_precision) -> int:
    '''Compare the kinds of two arrays of the same intrinsic type and rank.

    This is what tells apart the specifics of a generic interface that differ
    in nothing else -- LFRic's ``coordinate_jacobian`` has four, named for
    the real kind of their array arguments. Fortran resolves such a call by
    the kind of the actual, and so does this, as far as the PSyIR lets it:
    two kinds that reduce to different values are not a match at all, and two
    that cannot be reduced far enough to say are a weak match, so that a
    generic interface is refused as ambiguous rather than resolved by the
    order its specifics are named in.

    :param actual: one argument of the call.
    :param actual_precision: the kind of its elements.
    :type actual_precision: Optional[int |
        :py:class:`psyclone.psyir.symbols.ScalarType.Precision` |
        :py:class:`psyclone.psyir.nodes.DataNode`]
    :param dummy: the corresponding argument of the candidate routine.
    :param dummy_precision: the kind of its elements.
    :type dummy_precision: Optional[int |
        :py:class:`psyclone.psyir.symbols.ScalarType.Precision` |
        :py:class:`psyclone.psyir.nodes.DataNode`]

    :returns: :py:data:`EXACT_MATCH` where the two kinds are the same, or
        where one of the arrays has no kind to compare, and
        :py:data:`WEAK_MATCH` where they might or might not agree.

    :raises CallMatchingArgumentsNotFound: if the two kinds are both known
        and different.

    '''
    if actual_precision is None or dummy_precision is None:
        # An array of a derived type carries no kind, so there is nothing
        # here for the two to disagree about.
        return EXACT_MATCH
    actual_kind, actual_known = _reduced_kind(actual_precision, actual)
    dummy_kind, dummy_known = _reduced_kind(dummy_precision, actual)
    if actual_kind is None or dummy_kind is None:
        return WEAK_MATCH
    if actual_kind == dummy_kind:
        return EXACT_MATCH
    comparable = ((isinstance(actual_kind, int) and
                   isinstance(dummy_kind, int)) or
                  (isinstance(actual_kind, str) and
                   isinstance(dummy_kind, str)))
    if actual_known and dummy_known and comparable:
        raise CallMatchingArgumentsNotFound(
            f"Array kind mismatch of call argument "
            f"'{actual.debug_string().strip()}' (kind {actual_kind}) and "
            f"routine argument '{dummy.name}' (kind {dummy_kind})")
    return WEAK_MATCH


def _scalar_kinds_agree(actual: DataNode, actual_type, dummy_type) -> bool:
    '''Say whether two scalars of the same intrinsic type are written with
    kinds that reduce to the same thing.

    Two scalar types are equal only if their kinds are, and a kind written as
    a named constant is held as a reference to a symbol, so two declarations
    of the same kind under two names -- LFRic's ``r_def`` and ``r_double``
    are both ``real64`` -- are unequal types and the call was refused. The
    kinds are reduced by :py:func:`_reduced_kind`, the same way an array's
    are, and where they reduce to the same thing the two scalars are the same
    type after all.

    Only agreement is answered here. Two kinds that reduce to different
    things, and two that cannot be reduced, are both left to the type
    comparison the caller has already made, which refuses them: a scalar is
    passed by kind and a caller that got it wrong is not helped by a weak
    match.

    :param actual: one argument of the call.
    :param actual_type: the type of that argument.
    :type actual_type: :py:class:`psyclone.psyir.symbols.DataType`
    :param dummy_type: the type of the corresponding routine argument.
    :type dummy_type: :py:class:`psyclone.psyir.symbols.DataType`

    :returns: True if the two are scalars of the same intrinsic type whose
        kinds reduce to the same value or the same name.

    '''
    if not (isinstance(actual_type, ScalarType) and
            isinstance(dummy_type, ScalarType)):
        return False
    if actual_type.intrinsic != dummy_type.intrinsic:
        return False
    actual_kind, _ = _reduced_kind(actual_type.precision, actual)
    dummy_kind, _ = _reduced_kind(dummy_type.precision, actual)
    return actual_kind is not None and actual_kind == dummy_kind


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
      That one is exact: it relaxes nothing Fortran requires;
    * two arrays of the same intrinsic type and rank whose kinds cannot both
      be reduced to a value match weakly, since two names are not evidence of
      two values. Two kinds that *were* reduced and differ are not a match at
      all; see :py:func:`_match_kinds`.

    One match is not relaxed but widened: two scalars of the same intrinsic
    type whose kinds are written as different names for the same thing are
    equal types that the PSyIR held as unequal, and are an exact match; see
    :py:func:`_scalar_kinds_agree`. A scalar kind that cannot be reduced, or
    that reduces to something else, is refused as it was before.

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
        return _match_arrays(actual, actual_type, dummy, dummy_type,
                             interface_call)

    if isinstance(dummy_type, UnsupportedFortranType):
        return _match_partial(actual, actual_type, dummy, dummy_type,
                              interface_call)

    if actual_type != dummy_type:
        if (_type_symbols_match(actual_type, dummy_type) or
                _scalar_kinds_agree(actual, actual_type, dummy_type)):
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
