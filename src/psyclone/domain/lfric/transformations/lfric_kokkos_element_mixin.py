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


"""Pass one element of an array where the callee declares an array of one.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather
than instantiated. It holds no instance state and every method is a
``classmethod`` or a ``staticmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object. It sits beside
:py:class:`~psyclone.domain.lfric.transformations.lfric_kokkos_bound_mixin.\
LFRicKokkosBoundMixin`, whose :py:meth:`_inline_one` prepares a call by
calling :py:meth:`~LFRicKokkosElementMixin._section_element_actuals` among the
other preparation steps.

**Sequence association, and the one case of it this covers.** Fortran lets a
call pass ``a(k)`` -- one element -- where the callee declares an array, and
the callee then reads ``a(k)``, ``a(k+1)`` and so on up to its own declared
extent. PSyIR models arguments by rank, so such a call matches no routine at
all: ``CallMatchingArgumentsNotFound`` is raised before inlining begins, and
the region is refused for a call site that is perfectly legal Fortran.

Where the callee's extent is **one element**, sequence association says
nothing more than that the callee reads that one element, and the same call
written as the one-element section ``a(k:k)`` means exactly the same thing to
Fortran, to PSyIR and to the generated region. That rewriting is all this
mixin does. It is the shape LFRic's horizontal FFSL transport writes when it
treats one level at a time::

    do k = 1, nlayers
      field_local_tmp(1,:) = field_local(k,:)
      call monotonic_edge(field_local_tmp, mono_option(k), min_val,          &
                          field_edge_left(k), field_edge_right(k),           &
                          order+order_offset, 1, 1)

where ``monotonic_edge`` declares ``edge_left(nlayers)`` and the call passes
``nlayers`` as ``1``.

**The extent must be provable here, from this call.** The callee's declared
bound is read with this call's own actuals put in place of the dummies that
appear in it, and the rewriting is made only where both bounds are then
integer literals of the same value: one element, whatever the dummy was
called. Anything else -- an extent that is still a variable, a bound this
call does not determine, an element of a rank-2 array, an actual that is
already a section -- is left exactly as the kernel wrote it, and
:py:class:`~psyclone.psyir.transformations.InlineTrans` refuses it in its own
words with the call named. A rewriting that cannot be shown to cover one
element would be a silent narrowing of what the callee reads, which is the
one thing worse than a refusal.
"""

from psyclone.psyir.nodes import (
    ArrayReference, Literal, Range, Reference)
from psyclone.psyir.symbols import ArrayType, ScalarType


class LFRicKokkosElementMixin:
    """Let an element actual reach a callee that declares an array of one.

    **What is accepted.** A call with no named arguments, to a callee the
    Container holds under exactly one name, passing a plain subscripted
    element of a rank-1 array where the callee declares a rank-1 dummy whose
    declared bounds -- read with this call's actuals in place of the dummies
    they name -- are equal integer literals. The actual becomes the
    one-element section of itself and nothing else changes.

    **What is refused, by being left alone.** Every other pairing. A dummy
    whose extent is not fixed by this call, a dummy of more than one element,
    an actual that is an expression, a section, a whole array or an element of
    an array of rank two or more, and a callee name the Container holds more
    than one routine under, since which specific the call resolves to is
    PSyclone's to decide and not this mixin's to guess.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    @classmethod
    def _section_element_actuals(cls, call):
        """Rewrite each element actual of ``call`` that the callee reads whole.

        Made after the callee has been brought into the Container the call is
        made from, so that the callee's declarations can be read, and before
        :py:class:`~psyclone.psyir.transformations.InlineTrans` is asked to
        match the call against them.

        :param call: the call to prepare.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        """
        # The method the callees are found by is on the mixin this one sits
        # beside; the composed class resolves both.
        # pylint: disable=no-member
        routines = cls._local_callees(call)
        if len(routines) != 1:
            return
        if any(name is not None for name in call.argument_names):
            return
        formals = routines[0].symbol_table.argument_list
        positions = {id(formal): index for index, formal in enumerate(formals)}
        # Zipped rather than indexed: a call passing fewer actuals than the
        # callee takes dummies -- an optional argument left out -- pairs the
        # ones it does pass and stops.
        for formal, actual in zip(formals, list(call.arguments)):
            if not cls._is_element(actual):
                continue
            if not cls._holds_one_element(formal, call, positions):
                continue
            subscript = actual.indices[0]
            actual.replace_with(ArrayReference.create(
                actual.symbol,
                [Range.create(subscript.copy(), subscript.copy())]))

    @staticmethod
    def _is_element(actual):
        """Return whether ``actual`` is one subscripted element of an array.

        Matched by exact class rather than by ``isinstance``: a structure
        access is an ``ArrayReference``'s cousin and names a component this
        does not model. One subscript and a scalar type are between them the
        whole test -- an element of a rank-2 array carries two subscripts, a
        section carries a :py:class:`~psyclone.psyir.nodes.Range`, and an
        array whose declaration PSyclone could not model gives an unresolved
        type rather than a scalar one.

        :param actual: the actual argument to judge.
        :type actual: :py:class:`psyclone.psyir.nodes.Node`

        :returns: whether it subscripts a rank-1 array to one value.
        :rtype: bool
        """
        if type(actual) is not ArrayReference:  # pylint: disable=C0123
            return False
        if len(actual.indices) != 1 or actual.walk(Range):
            return False
        return isinstance(actual.datatype, ScalarType)

    @classmethod
    def _holds_one_element(cls, formal, call, positions):
        """Return whether ``formal`` declares exactly one element at ``call``.

        :param formal: the dummy argument whose declaration is read.
        :type formal: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param call: the call whose actuals fix the declaration's bounds.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param positions: the position in the argument list of each dummy,
            by the identity of its symbol.
        :type positions: Dict[int, int]

        :returns: whether the dummy is a rank-1 array whose declared bounds,
            with this call's actuals in place of the dummies they name, are
            integer literals of one and the same value.
        :rtype: bool
        """
        datatype = formal.datatype
        if not isinstance(datatype, ArrayType) or len(datatype.shape) != 1:
            return False
        bounds = datatype.shape[0]
        if not isinstance(bounds, ArrayType.ArrayBounds):
            return False
        lower = cls._element_bound_value(
            cls._with_actuals(bounds.lower, call, positions))
        upper = cls._element_bound_value(
            cls._with_actuals(bounds.upper, call, positions))
        return lower is not None and lower == upper

    @staticmethod
    def _element_bound_value(bound):
        """Return the value of ``bound`` where it is an integer literal.

        Named for the one thing it reads, and at length, because the mixins
        of :py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans`
        share one namespace: ``_literal_value`` is already
        :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_call_mixin.LFRicKokkosCallMixin._literal_value`, which folds a
        Python ``ast`` rather than PSyIR, and a second method of that name
        here would be silently overridden by whichever mixin the composed
        class lists first -- returning ``None`` for every bound, from a method
        that is never entered, with nothing in the traceback to say so.

        Arithmetic is read and not folded: ``value(n + 1)`` called with
        ``n`` as zero declares one element, and this answers ``None`` for it.
        Folding would need a constant evaluator, and every bound a kernel
        writes for a single level is written as the size itself.

        :param bound: a substituted bound of a dummy's declared shape.
        :type bound: :py:class:`psyclone.psyir.nodes.Node`

        :returns: the value, or ``None`` where the bound is anything but an
            integer literal.
        :rtype: Optional[int]
        """
        if (isinstance(bound, Literal)
                and bound.datatype.intrinsic is ScalarType.Intrinsic.INTEGER):
            return int(bound.value)
        return None

    @staticmethod
    def _with_actuals(bound, call, positions):
        """Return ``bound`` with this call's actuals for the dummies it names.

        The bound is copied first, so the callee's own declaration is read
        and not rewritten: the callee may be called from more than one site
        and each reads it afresh.

        :param bound: one bound of a dummy's declared shape.
        :type bound: :py:class:`psyclone.psyir.nodes.Node`
        :param call: the call whose actuals are substituted.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param positions: the position in the argument list of each dummy,
            by the identity of its symbol.
        :type positions: Dict[int, int]

        :returns: the bound with every reference to a dummy this call passes
            an actual for replaced by that actual.
        :rtype: :py:class:`psyclone.psyir.nodes.Node`
        """
        bound = bound.copy()
        if isinstance(bound, Reference):
            index = positions.get(id(bound.symbol))
            if index is not None and index < len(call.arguments):
                return call.arguments[index].copy()
            return bound
        for reference in bound.walk(Reference):
            index = positions.get(id(reference.symbol))
            if index is not None and index < len(call.arguments):
                reference.replace_with(call.arguments[index].copy())
        return bound


__all__ = ["LFRicKokkosElementMixin"]
