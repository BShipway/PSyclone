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

"""Choose the implementation to capture, and put its body into the shape
the region is described from.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

Everything here is asked of the *schedule* rather than of an argument, a type
or a bound, and the three questions run in that order:

* **Which implementation is being captured?** A kind-polymorphic kernel is one
  name over several specific procedures, so :py:meth:`\
LFRicKokkosScheduleMixin._schedule` picks the one the algorithm layer's
  precisions select and refuses rather than guesses when it cannot;
  :py:meth:`LFRicKokkosScheduleMixin._matches` is the per-candidate question,
  and it is PSyclone's own rather than this transformation's.
* **What shape is the body in?** :py:meth:`\
LFRicKokkosScheduleMixin._lower_sections` rewrites every array-valued
  assignment into an explicit loop, because the generated region has no way to
  say ``a(i:j)``.
* **Which loops may be spread over the team?** :py:meth:`\
LFRicKokkosScheduleMixin._parallel_loops` asks
  :py:class:`~psyclone.psyir.tools.DependencyTools` and then narrows what it
  accepts by three properties of the shape the loop is rendered into and one
  of what spreading it would cost.

It is a module of its own because
:py:mod:`psyclone.domain.lfric.transformations.lfric_kokkos_trans` is mostly
the capture contract the user guide publishes -- some six hundred lines of
class docstring -- and the steps below are the transformation's *method*
rather than its contract. Nothing here is named in that contract except
:py:meth:`LFRicKokkosScheduleMixin._parallel_loops`, which the team-launch
paragraph points at.

The one constraint that follows is that a method reaching a helper of a
sibling mixin does so through ``cls``, resolved on ``LFRicKokkosTrans``:
:py:meth:`LFRicKokkosScheduleMixin._lower_sections` asks
``cls._is_array_valued``, which is ``LFRicKokkosContractMixin``'s. Calling
that method directly on this mixin is therefore not supported.
"""

from psyclone.errors import GenerationError
from psyclone.psyir.nodes import (
    Assignment, Exit, Literal, Loop, WhileLoop)
from psyclone.psyir.nodes.array_mixin import ArrayMixin
from psyclone.psyir.symbols import ScalarType
from psyclone.psyir.tools import DependencyTools
from psyclone.psyir.transformations import (
    ArrayAssignment2LoopsTrans, Reference2ArrayRangeTrans,
    TransformationError)


class LFRicKokkosScheduleMixin:
    """Select the kernel schedule to capture and prepare it for description.

    What the region declares for each argument is
    ``LFRicKokkosArgumentMixin``; what a symbol is in C terms is
    ``LFRicKokkosTypesMixin``; the refusals that decide whether any of this is
    reached at all are ``LFRicKokkosContractMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The fewest iterations a loop of constant trip count must have to be
    #: spread over the team. Spreading a loop is what makes a region
    #: hierarchical, and a hierarchical team is at least a warp: every array
    #: the loop names becomes team scratch, and every write to a shared array
    #: outside a spread loop becomes a ``Kokkos::single`` and a
    #: ``team_barrier`` that all its members wait at. A loop over the three
    #: components of a vector pays all of that to occupy three members of
    #: thirty-two. Measured on an H100
    #: (phase 8, task C1) ``nodal_xyz_coordinates_code``, whose only spread
    #: loops were such loops, ran its column on one member of each team
    #: through 73 ``single`` blocks and took 48% of all card time at C144.
    #: Left serial, such a loop is run by each member for its own cell, and
    #: the arrays it names become the member's own. Eight is a quarter of a
    #: warp: below it a spread loop leaves most of the team idle whatever the
    #: body, and the loops GungHo has of constant count are of three or four.
    SPREAD_MIN_TRIPS = 8

    @classmethod
    def _schedule(cls, kernel):
        """Return the PSyIR schedule of the kernel to be captured.

        A kind-polymorphic kernel resolves to one schedule per specific
        procedure of its generic interface, and the one to capture is the one
        Fortran would have resolved the call to. That question is
        :py:meth:`~psyclone.domain.lfric.LFRicKern.validate_kernel_code_args`'s
        rather than this transformation's; see :py:meth:`_matches`. A
        built-in has no file and so no callee: its schedule is synthesised
        by :py:meth:`LFRicKokkosBuiltinMixin._builtin_schedule`.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :returns: the kernel's schedule, which
            :py:meth:`~psyclone.domain.lfric.LFRicKern.get_callees` caches so
            that transformations applied to it persist.
        :rtype: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if no schedule matches the precisions the
            algorithm layer passes, if more than one does, or if the matcher
            cannot model the kernel's metadata and so cannot answer at all.
        """
        if cls._is_builtin(kernel):
            # A built-in has no file for get_callees() to read; its schedule
            # is written for it, see LFRicKokkosBuiltinMixin.
            return cls._builtin_schedule(kernel)
        schedules = kernel.get_callees()
        if len(schedules) == 1:
            return schedules[0]
        try:
            matches = [schedule for schedule in schedules
                       if cls._matches(kernel, schedule)]
        except NotImplementedError as err:
            # The matcher builds the interface the metadata implies before it
            # compares anything, and PSyclone's issue #928 leaves parts of that
            # unbuilt -- stencils and CMA kernels, the evaluator and
            # inter-grid shapes having since been described.
            # Not being able to ask the question is a third outcome,
            # distinct from asking it and getting no match: reading it as one
            # would report a kind mismatch about a kernel whose kinds were
            # never examined.
            raise TransformationError(
                f"LFRicKokkosTrans cannot tell which of the "
                f"{len(schedules)} implementations of '{kernel.name}' the "
                f"algorithm layer calls: the metadata is outside what "
                f"PSyclone's own matcher models ({err}).") from err
        if not matches:
            raise TransformationError(
                f"LFRicKokkosTrans found no implementation of "
                f"'{kernel.name}' matching the precisions the algorithm layer "
                f"passes, out of {len(schedules)}.")
        if len(matches) > 1:
            names = ", ".join(sorted(schedule.name for schedule in matches))
            raise TransformationError(
                f"LFRicKokkosTrans found {len(matches)} implementations of "
                f"'{kernel.name}' matching the precisions the algorithm layer "
                f"passes ({names}), and will not choose between them.")
        return matches[0]

    @staticmethod
    def _matches(kernel, schedule):
        """Say whether one implementation matches the algorithm's precisions.

        The question is PSyclone's own, not this transformation's:
        :py:meth:`~psyclone.domain.lfric.LFRicKern.validate_kernel_code_args`
        exists, by its own comment, "to identify the correct kernel subroutine
        for a mixed-precision kernel". It converts both the formal and the
        algorithm-layer kinds to byte widths through the LFRic configuration's
        ``precision_map`` and raises when they disagree.

        Matching in widths rather than in kind names is right for a back-end
        that emits a width, and is why two implementations can both match:
        ``precision_map`` gives ``r_single`` and ``r_solver`` the same 4 bytes
        where Fortran, resolving by name, keeps them apart.
        :py:meth:`_schedule` refuses that case rather than picking one.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`
        :param schedule: one of the kernel's candidate implementations.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: whether the algorithm layer could have called this one.
        :rtype: bool

        :raises NotImplementedError: if the matcher cannot build the interface
            the kernel's metadata implies. Deliberately not caught here:
            :py:meth:`_schedule` turns it into a refusal, because a matcher
            that cannot answer has not answered "no".
        """
        try:
            kernel.validate_kernel_code_args(schedule.symbol_table)
        except GenerationError:
            return False
        return True

    @classmethod
    def _lower_sections(cls, schedule):
        """Replace every array-valued assignment with an explicit loop.

        Applied to the schedule :py:meth:`_schedule` returns, which
        :py:meth:`~psyclone.domain.lfric.LFRicKern.get_callees` caches so that
        transformations applied to a kernel persist. That is the intended
        idiom, so the lowering is done once here rather than repeated for
        every consumer of the schedule.

        Two shapes reach the lowering and one is kept from it. A written
        section, ``a(2:n) = 0.0``, is what
        :py:class:`~psyclone.psyir.transformations.ArrayAssignment2LoopsTrans`
        exists for. A whole array named with no accessor at all,
        ``pv_at_quad = 0.0``, is the same statement written the shorter way,
        and that transformation refuses it for want of an accessor; writing
        one in with
        :py:class:`~psyclone.psyir.transformations.Reference2ArrayRangeTrans`
        first makes it the section it already meant. The exception is an
        array constructor, ``cells(:) = [2, 3, 4, 5]``, whose values are
        positional: :py:class:`~psyclone.psyir.backend.c.CWriter` renders one
        element by element from the constructor itself, and a loop would
        leave behind a subscripted constructor that no writer can render.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if an assignment holding a section
            cannot be lowered. :py:meth:`_validate_sections` predicts this by
            running this method over a copy, so reaching it from
            :py:meth:`apply` would mean that prediction had been skipped.
        """
        # pylint: disable=no-member
        cls._hoist_intrinsic_operands(schedule)
        cls._hoist_constructors(schedule)
        # pylint: enable=no-member
        expansion = Reference2ArrayRangeTrans()
        lowering = ArrayAssignment2LoopsTrans()
        for assignment in schedule.walk(Assignment):
            if not cls._is_array_valued(assignment):
                continue
            if not isinstance(assignment.lhs, ArrayMixin):
                expansion.apply(assignment.lhs)
            lowering.apply(assignment)

    @staticmethod
    def _constant_trips(loop):
        """Return a loop's trip count where its bounds are integer literals.

        The bounds are read after the kernel's named constants have been
        substituted, so ``do i = 1, 3`` and a loop to a ``parameter`` of 3
        both answer 3; a bound naming anything else answers ``None``.

        :param loop: the loop being asked about, with a step of one.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`

        :returns: the number of iterations, or ``None`` if not constant.
        :rtype: Optional[int]
        """
        bounds = (loop.start_expr, loop.stop_expr)
        if not all(isinstance(bound, Literal) and
                   bound.datatype.intrinsic ==
                   ScalarType.Intrinsic.INTEGER for bound in bounds):
            return None
        return max(int(bounds[1].value) - int(bounds[0].value) + 1, 0)

    @classmethod
    def _parallel_loops(cls, schedule):
        """Return the loops of ``schedule`` that may be spread over the team.

        The judgement is PSyclone's own:
        :py:meth:`~psyclone.psyir.tools.DependencyTools.\
can_loop_be_parallelised`
        is what decides whether a loop's iterations are independent, so a
        recurrence such as ``x_new(k + 1) = ... x_new(k) ...`` is left where it
        is rather than being re-analysed here. It is conservative in a way that
        matters for LFRic: a write through a dofmap, ``field(map(df) + k)``,
        reads as a write-write race because the indirection is opaque to it,
        so a kernel whose only loops write that way keeps the flat launch.

        Four rules narrow what it accepts, the first three properties of the
        shape the loop is rendered into rather than of the dependence
        analysis, the last one of its cost:

        * **Outermost wins.** A team is one pool of members, so nesting a
          ``TeamVectorRange`` inside another would divide the same members
          twice. A loop with a chosen ancestor is therefore skipped, which
          leaves the outermost of any parallelisable nest.
        * **A stepped loop is skipped.** ``TeamVectorRange(team, begin, end)``
          counts by one and has no stride, so a loop that does not is left as
          a serial ``for`` even where the analysis would allow it.
        * **A loop an EXIT leaves is skipped.** The body of a spread loop is
          a lambda, and C++ has no break that leaves the loop the lambda was
          launched over. The loop an :py:class:`~psyclone.psyir.nodes.Exit`
          names -- its innermost enclosing loop -- is therefore left serial,
          where the break is what the Fortran meant. An EXIT further in
          leaves a loop of its own inside the lambda and does not disqualify
          anything.
        * **A short loop of constant count is skipped.** A loop whose
          bounds are integer literals and which runs fewer than
          :py:attr:`SPREAD_MIN_TRIPS` times stays a serial ``for``: it
          would occupy a few members of a warp-sized team, put the arrays
          it names in team scratch, and make every write to them outside
          it a ``single`` and a barrier. The loop over
          ``nlayers`` is never skipped this way, because its count is not a
          literal. A region whose only candidates are skipped spreads
          nothing, and is launched one cell per team rank. In a region that
          spreads other loops, a short loop is spread after all if it
          assigns an array the team shares: serial, each iteration's write
          would be a ``single`` and a barrier of its own. Leaving one loop
          serial can make another's array shared, so the choice is repeated
          until it settles.

        :param schedule: the kernel schedule being captured, already lowered
            and bound-substituted, since both create loops.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: the loops to spread, outermost first, in schedule order.
        :rtype: Tuple[:py:class:`psyclone.psyir.nodes.Loop`, ...]
        """
        tools = DependencyTools()
        serial = None
        while True:
            chosen, short = cls._select_loops(schedule, tools, serial)
            if serial is None:
                serial = short
            if not chosen:
                return ()
            shared = [loop for loop in serial
                      if any(loop is each for each in short)
                      and not cls._writes_only_member_local(
                          schedule, loop, chosen)]
            if not shared:
                return tuple(chosen)
            serial = [loop for loop in serial
                      if not any(loop is each for each in shared)]

    @classmethod
    def _select_loops(cls, schedule, tools, serial):
        """Walk ``schedule`` once, choosing the loops to spread.

        The rules are :py:meth:`_parallel_loops`'s. A short loop of constant
        count is left serial when ``serial`` is ``None``, which is the first
        walk, or when it is one of ``serial``; otherwise it is judged like
        any other loop.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param tools: the dependence analysis, created once by the caller.
        :type tools: :py:class:`psyclone.psyir.tools.DependencyTools`
        :param serial: the short loops still to be left serial, or ``None``
            to leave every one of them so.
        :type serial: Optional[List[:py:class:`psyclone.psyir.nodes.Loop`]]

        :returns: the loops chosen, outermost first, and the short loops the
            walk reached, would otherwise have chosen and left serial.
        :rtype: Tuple[List[:py:class:`psyclone.psyir.nodes.Loop`],
            List[:py:class:`psyclone.psyir.nodes.Loop`]]
        """
        chosen = []
        short = []
        for loop in schedule.walk(Loop):
            ancestor = loop.ancestor(Loop)
            nested = False
            while ancestor is not None:
                if any(ancestor is entry for entry in chosen):
                    nested = True
                    break
                ancestor = ancestor.ancestor(Loop)
            if nested:
                continue
            step = loop.step_expr
            if not (isinstance(step, Literal) and step.value == "1"):
                continue
            if any(statement.ancestor((Loop, WhileLoop)) is loop
                   for statement in loop.walk(Exit)):
                continue
            if not tools.can_loop_be_parallelised(loop):
                continue
            trips = cls._constant_trips(loop)
            if (trips is not None and trips < cls.SPREAD_MIN_TRIPS and
                    (serial is None or
                     any(loop is each for each in serial))):
                short.append(loop)
                continue
            chosen.append(loop)
        return chosen, short

    @classmethod
    def _writes_only_member_local(cls, schedule, loop, chosen):
        """Say whether every array ``loop`` assigns is a member's own.

        Where a region spreads anything, an array the team shares is written
        outside the spread loops under a ``Kokkos::single`` and a barrier.
        A short loop left serial that writes one pays those once per
        iteration where spread it paid one barrier, so it is left serial
        only where every array it assigns is a kernel-local that
        :py:meth:`~LFRicKokkosCallMixin._is_member_local` gives each member,
        judged against the loops ``chosen`` for the team.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param loop: the short loop being judged.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param chosen: the loops to be spread over the team.
        :type chosen: List[:py:class:`psyclone.psyir.nodes.Loop`]

        :returns: whether the loop writes no array the team shares.
        :rtype: bool
        """
        # pylint: disable=no-member
        locals_ = schedule.symbol_table.automatic_datasymbols
        aliases = cls._alias_targets(schedule)
        targets = {target for aimed in aliases.values() for target in aimed}
        for assignment in loop.walk(Assignment):
            symbol = assignment.lhs.symbol
            if not symbol.is_array:
                continue
            if not any(symbol is each for each in locals_):
                return False
            try:
                extents = cls._extents(symbol)
            except TransformationError:
                return False
            if not cls._is_member_local(
                    symbol, extents, tuple(chosen), targets):
                return False
        return True


__all__ = ["LFRicKokkosScheduleMixin"]
