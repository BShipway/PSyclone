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

Three things live here, and the first two are the same thing seen from either
end.

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

:py:meth:`LFRicKokkosIntrinsicMixin._lower_reductions` converts in the same
direction as the allocations do, and for the same reason: it moves a statement
into the one shape every later rule already knows. The backend writes ``SUM``,
``MINVAL``, ``MAXVAL`` and ``DOT_PRODUCT`` as a bounded loop over a scalar
accumulator, and writes them on the right-hand side of an assignment and
nowhere else, because a loop is not an expression in C++ and has to be placed
*ahead* of the statement that reads its value. A fold standing anywhere else
-- ``if (MAXVAL(switch(low:high)) > 0)``, which is how the FFSL
departure-point kernels ask whether a sweep found any cell -- is given an
assignment of its own immediately before the statement it stood in, and the
call is replaced by a reference to the scalar that assignment writes. What was
refused twice over, once for the fold the writer had no spelling for in that
position and once for the section standing outside an assignment, is then the
statement pair the transformation already lowers.
"""

from psyclone.psyir.backend.kokkos import KokkosWriter
from psyclone.psyir.backend.kokkos_array_intrinsics import (
    KokkosArrayIntrinsics)
from psyclone.psyir.nodes import (
    ArrayReference, Assignment, IntrinsicCall, Loop, Range, Reference,
    Schedule, WhileLoop)
from psyclone.psyir.symbols import ArrayType, DataSymbol, ScalarType
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

    #: The root name of the scalar a hoisted fold leaves its value in. The
    #: intrinsic's own name goes in front of it, so a reader of the generated
    #: region meets ``maxval_result`` where the Fortran read ``MAXVAL``, and
    #: ``SymbolTable.new_symbol`` numbers a second one rather than colliding
    #: with the first or with anything the kernel declared.
    _FOLD_RESULT = "result"

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

        The writer is asked twice, because an intrinsic can be refused for
        two different reasons and only one of them is a missing spelling. A
        handler has no spelling for ``TINY``; the array-valued tier has every
        spelling it needs for ``RESHAPE`` and still cannot generate one whose
        source it cannot take the shape of. The first probe steps over a tier
        intrinsic where the tier writes it -- no handler does -- so the
        second asks the tier there, and between them nothing this
        transformation accepts is left for the back-end to refuse.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if the Kokkos writer, asked with the
            kinds this region would be generated with, refuses any intrinsic
            the body reads, or cannot take the shape of an array expression
            it holds.
        """
        kind_types = cls._kind_types(schedule)
        refused = KokkosWriter().unsupported_intrinsics(schedule, kind_types)
        if refused:
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture a kernel whose body reads "
                f"{', '.join(refused)}: the Kokkos writer has no spelling "
                "for it, and a region it cannot write is refused here rather "
                "than by the back-end.")
        unshapeable = KokkosWriter().unshapeable_expressions(
            schedule, kind_types)
        if unshapeable:
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture this kernel: "
                f"{' '.join(unshapeable)} The refusal is the Kokkos writer's "
                "own, asked here rather than left to the back-end.")

    @classmethod
    def _lower_reductions(cls, schedule):
        """Give every fold outside an assignment a scalar of its own.

        ``SUM``, ``MINVAL``, ``MAXVAL`` and ``DOT_PRODUCT`` are written by
        :py:class:`~psyclone.psyir.backend.kokkos_array_intrinsics.\
KokkosArrayIntrinsics`
        as a bounded loop over an accumulator, placed ahead of the statement
        that reads the accumulator, because a loop is not an expression in
        C++. The one position that offers such a place is the right-hand side
        of an assignment, and the writer says so itself in
        :py:meth:`~psyclone.psyir.backend.kokkos_intrinsics_mixin.\
KokkosIntrinsicsMixin.written_by_the_array_tier`, which is the question asked
        here rather than a second copy of it.

        A fold anywhere else is moved rather than refused. ``if
        (MAXVAL(switch(low:high)) > 0)`` becomes ``maxval_result =
        MAXVAL(switch(low:high))`` immediately before the ``if``, and the
        condition reads the scalar. Immediately before, and not hoisted any
        further: the operands of these folds are the running arrays of a
        sweep, so a statement moved past another that writes one of them would
        fold different values. Placing it in the statement's own position
        keeps the order the Fortran states, and with it the bit-exactness of
        a one-thread host build, since the loop the writer generates runs the
        section's own indices in the section's own order.

        Only a fold whose *value* is a scalar is moved. ``MATMUL``,
        ``TRANSPOSE``, ``RESHAPE`` and a ``SUM`` with a ``dim`` produce
        arrays, which would need a temporary of their own shape: those stay
        where they are and keep whatever refusal they already had.

        The position is read from the innermost node the enclosing
        :py:class:`~psyclone.psyir.nodes.Schedule` holds, rather than from
        :py:meth:`~psyclone.psyir.nodes.Node.ancestor` of
        :py:class:`~psyclone.psyir.nodes.Statement`: a
        :py:class:`~psyclone.psyir.nodes.Call` is itself a ``Statement``, so
        the ancestor of the ``MAXVAL`` in ``ABS(MAXVAL(a))`` is the ``ABS``,
        which has no place in a schedule to insert anything before.

        :param schedule: the kernel schedule being captured, rewritten in
            place.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if the fold stands in the condition of a
            ``DO WHILE``, where the value is re-read on every trip and a
            statement before the loop would be evaluated once.
        """
        # One fold at a time, re-walking after each: the statement a fold is
        # moved into carries copies of everything that was inside it, and the
        # originals leave the tree with it, so a list taken once would hold
        # nodes that are no longer part of the schedule. Each pass strictly
        # reduces what is left, because a moved fold -- and every fold inside
        # it -- is one the tier now writes where it stands.
        while True:
            pending = [call for call in schedule.walk(IntrinsicCall)
                       if cls._is_hoisted_fold(call)]
            if not pending:
                return
            cls._hoist_fold(schedule, pending[0])

    @staticmethod
    def _is_hoisted_fold(call):
        """Answer whether this call is a fold standing where no loop can go.

        :param call: the intrinsic call to judge.
        :type call: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: whether :py:meth:`_lower_reductions` moves it.
        :rtype: bool
        """
        return (KokkosArrayIntrinsics.handles(call)
                and isinstance(call.datatype, ScalarType)
                and not KokkosWriter.written_by_the_array_tier(call))

    @classmethod
    def _hoist_fold(cls, schedule, call):
        """Move one fold into an assignment of its own.

        :param schedule: the kernel schedule holding the fold, whose symbol
            table declares the scalar.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param call: the fold to move.
        :type call: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :raises TransformationError: if the fold stands in the condition of a
            ``DO WHILE``.
        """
        statement = call
        while not isinstance(statement.parent, Schedule):
            statement = statement.parent
        if isinstance(statement, WhileLoop) and any(
                node is call
                for node in statement.condition.walk(IntrinsicCall)):
            condition = statement.condition.debug_string().strip()
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture "
                f"'{call.debug_string().strip()}' where it stands, in the "
                f"condition 'do while ({condition})': the generated region "
                "computes a fold in a loop of its own, which would stand "
                "before the while loop and be evaluated once where Fortran "
                "evaluates the condition on every trip.")
        symbol = schedule.symbol_table.new_symbol(
            f"{call.intrinsic.name.lower()}_{cls._FOLD_RESULT}",
            symbol_type=DataSymbol, datatype=call.datatype)
        statement.parent.children.insert(
            statement.position,
            Assignment.create(Reference(symbol), call.copy()))
        call.replace_with(Reference(symbol))

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
