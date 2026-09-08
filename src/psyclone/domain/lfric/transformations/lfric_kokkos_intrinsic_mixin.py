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

"""Answer what the writers can spell, and convert what they cannot.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``classmethod``, which is what makes the mixin sound: it is a namespace with an
inheritable ``cls``, not an object.

Two things live here, and they are the same thing seen from either end.

:py:meth:`LFRicKokkosIntrinsicMixin._validate_intrinsics` closes the gap
between what ``validate`` accepted and what the writers could write.
``validate`` used to judge a body by its *shape* -- its loops, its sections,
its locals -- and leave the writers to discover, well after the capture had
been accepted, that they had no spelling for one of its intrinsics. That
surfaced as a back-end error out of ``apply``, which is exactly what a
transformation's ``validate`` exists to prevent. The check asks the writer
itself rather than keeping a second list beside it, so the two cannot part
company as either grows.

:py:meth:`LFRicKokkosIntrinsicMixin._lower_allocations` converts the other
direction. A kernel-local ``allocatable`` sized at entry from the kernel's own
arguments and freed before the routine returns is the automatic array the
region already places in team scratch, written the other way round: the shape
is stated by an ``ALLOCATE`` instead of by the declaration. Rewriting the
declaration from the ``ALLOCATE`` and removing both statements makes it one,
after which every rule about a local array -- its kind, its extents, its
scratch description -- applies unchanged and none of them knows it was ever an
allocation.

What that conversion may not do is the reason it is a conversion rather than a
translation. The launch computes its scratch size on the host, before it enters
the region, where the only values in scope are the region's own scalars: an
extent that is a reduction over the kernel's data is not one, and an allocation
inside a loop is not one array with a size at all. Both are refused by name.
"""

from psyclone.psyir.backend.kokkos import KokkosWriter
from psyclone.psyir.backend.kokkos_array_intrinsics import (
    KokkosArrayIntrinsics)
from psyclone.psyir.nodes import (
    ArrayReference, IntrinsicCall, Loop, Range, Reference)
from psyclone.psyir.symbols import ArrayType
from psyclone.psyir.transformations.transformation_error import (
    TransformationError)


class LFRicKokkosIntrinsicMixin:
    """Judge and convert the intrinsics a captured kernel body reads."""
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The two statements the allocation tier converts. Neither is an
    #: expression: both are whole statements the generated region removes,
    #: rather than anything a writer spells.
    _ALLOCATIONS = (IntrinsicCall.Intrinsic.ALLOCATE,
                    IntrinsicCall.Intrinsic.DEALLOCATE)

    #: Arguments of an ``ALLOCATE`` that say something beyond the shape.
    #: ``source`` and ``mold`` state the value or the type as well, and
    #: ``stat`` and ``errmsg`` ask about a failure the region cannot have,
    #: scratch being reserved by the launch and never by the body. The
    #: frontend upper-cases the keyword it read, so the comparison folds
    #: case rather than assuming either spelling.
    _ALLOCATE_OPTIONS = ("stat", "errmsg", "source", "mold")

    @staticmethod
    def _written_as_a_nest(assignment):
        """Answer whether the backend writes this assignment as it stands.

        The section lowering rewrites ``a(:) = ...`` into a PSyIR loop before
        any writer sees it, and
        :py:class:`~psyclone.psyir.transformations.ArrayAssignment2LoopsTrans`
        takes only a right-hand side that is scalar-valued or elemental. A
        contraction is neither: ``exner_e(:) = MATMUL(m, rhs_e)`` -- the shape
        ``set_exner_code`` and ``set_rho_code`` are written in -- is exactly
        what that transformation declines. Refusing on the decline would refuse
        the shape this tier exists to write, so such an assignment is kept from
        the lowering and left to
        :py:class:`~psyclone.psyir.backend.kokkos_array_intrinsics.\
KokkosArrayIntrinsics`, which generates the nest over the destination
        directly.

        :param assignment: the assignment to judge.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`

        :returns: whether the backend writes it without the section lowering.
        :rtype: bool
        """
        return KokkosArrayIntrinsics.holds(assignment.rhs)

    @classmethod
    def _validate_intrinsics(cls, schedule):
        """Refuse a body naming an intrinsic the writers cannot spell.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if the Kokkos writer, asked with the
            kinds this region would be generated with, refuses any intrinsic
            the body reads.
        """
        refused = KokkosWriter().unsupported_intrinsics(
            schedule, cls._kind_types(schedule))
        if refused:
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture a kernel whose body reads "
                f"{', '.join(refused)}: the Kokkos writer has no spelling "
                "for it, and a region it cannot write is refused here rather "
                "than by the back-end.")

    @classmethod
    def _lower_allocations(cls, schedule):
        """Turn an allocated kernel-local into a declared one.

        The declaration is rewritten from the ``ALLOCATE``'s own bounds and
        both statements are removed, so that what is left is the automatic
        array the rest of the transformation already knows how to place. The
        rewriting is done before anything is removed, because the bounds are
        read out of the statement being removed, and after every statement has
        been read, because rewriting a declaration is what stops it being an
        allocatable and the ``DEALLOCATE`` is still to be judged.

        :param schedule: the kernel schedule being captured, rewritten in
            place.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: as :py:meth:`_allocated` does.
        """
        calls = [call for call in schedule.walk(IntrinsicCall)
                 if call.intrinsic in cls._ALLOCATIONS]
        allocated = set()
        shapes = []
        for call in calls:
            shapes.extend(cls._allocated(call, allocated))
        for symbol, reference in shapes:
            symbol.datatype = ArrayType(
                symbol.datatype.elemental_type, cls._shape(reference))
        for call in calls:
            call.detach()

    @classmethod
    def _allocated(cls, call, allocated):
        """Return the arrays one allocation statement names, checking it.

        :param call: the ``ALLOCATE`` or ``DEALLOCATE`` to read.
        :type call: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        :param allocated: the arrays already allocated by an earlier
            statement, added to by this call.
        :type allocated: set[str]

        :returns: the symbol and the reference of each array an ``ALLOCATE``
            gives a shape, and nothing for a ``DEALLOCATE``.
        :rtype: list[tuple[:py:class:`psyclone.psyir.symbols.DataSymbol`,
            :py:class:`psyclone.psyir.nodes.ArrayReference`]]

        :raises TransformationError: if the statement is inside a loop, if it
            carries an argument beyond the arrays it names, if it names
            something that is not a kernel-local allocatable, or if it
            allocates an array a previous statement had already allocated.
        """
        names = ", ".join(f"'{cls._allocated_name(argument)}'"
                          for argument in call.arguments)
        if call.ancestor(Loop) is not None:
            raise TransformationError(
                f"LFRicKokkosTrans needs the allocation of {names} to be "
                "outside every loop of the kernel: the generated region "
                "reserves its scratch once, before the launch, where an "
                "allocation made afresh on each trip is a different array "
                "each time.")
        for name in call.argument_names:
            if name is not None and name.lower() in cls._ALLOCATE_OPTIONS:
                raise TransformationError(
                    f"LFRicKokkosTrans cannot convert the allocation of "
                    f"{names} to scratch: it carries a '{name.lower()}' "
                    "argument, which says something the reserved scratch "
                    "does not carry.")
        arrays = []
        for argument in call.arguments:
            symbol = cls._allocatable(call, argument, names)
            if call.intrinsic is IntrinsicCall.Intrinsic.DEALLOCATE:
                continue
            if symbol.name in allocated:
                raise TransformationError(
                    f"LFRicKokkosTrans cannot convert the allocation of "
                    f"'{symbol.name}' to scratch: it is allocated more than "
                    "once, and scratch is reserved with one shape for the "
                    "whole region.")
            allocated.add(symbol.name)
            arrays.append((symbol, argument))
        return arrays

    @classmethod
    def _allocatable(cls, call, argument, names):
        """Return the local allocatable one argument of a statement names.

        :param call: the statement the argument belongs to.
        :type call: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        :param argument: the argument to read.
        :type argument: :py:class:`psyclone.psyir.nodes.Node`
        :param str names: the arrays the statement names, for the refusal.

        :returns: the symbol allocated or deallocated.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :raises TransformationError: if the argument is not a reference to a
            kernel-local allocatable array, or if an ``ALLOCATE`` gives it no
            explicit shape.
        """
        symbol = getattr(argument, "symbol", None)
        datatype = getattr(symbol, "datatype", None)
        if symbol is None or not isinstance(datatype, ArrayType) \
                or not datatype.is_allocatable or not symbol.is_automatic:
            raise TransformationError(
                f"LFRicKokkosTrans cannot convert the allocation of {names} "
                f"to scratch: '{argument.debug_string().strip()}' is not a "
                "kernel-local allocatable array, so it is not the automatic "
                "array an allocated local otherwise is.")
        if call.intrinsic is IntrinsicCall.Intrinsic.ALLOCATE and not (
                isinstance(argument, ArrayReference)
                and all(isinstance(index, Range)
                        for index in argument.indices)):
            raise TransformationError(
                f"LFRicKokkosTrans cannot convert the allocation of {names} "
                f"to scratch: '{argument.debug_string().strip()}' states no "
                "explicit shape, so there is no size to reserve.")
        return symbol

    @staticmethod
    def _allocated_name(argument):
        """Return the name one argument of an allocation statement carries.

        :param argument: the argument to name.
        :type argument: :py:class:`psyclone.psyir.nodes.Node`

        :returns: the array's name, or the argument as written where it names
            no array at all.
        :rtype: str
        """
        if isinstance(argument, Reference):
            return argument.symbol.name
        return argument.debug_string().strip()

    @staticmethod
    def _shape(reference):
        """Return the declared shape an ``ALLOCATE`` states for one array.

        The bounds are copied out of the statement rather than referred to,
        because the statement is about to be removed and the datatype they
        are written into outlives it.

        :param reference: the array as the ``ALLOCATE`` names it.
        :type reference: :py:class:`psyclone.psyir.nodes.ArrayReference`

        :returns: one ``(lower, upper)`` bound pair per dimension, as the
            declaration would have carried.
        :rtype: list[tuple[:py:class:`psyclone.psyir.nodes.DataNode`,
            :py:class:`psyclone.psyir.nodes.DataNode`]]
        """
        return [(index.start.copy(), index.stop.copy())
                for index in reference.indices]


__all__ = ["LFRicKokkosIntrinsicMixin"]
