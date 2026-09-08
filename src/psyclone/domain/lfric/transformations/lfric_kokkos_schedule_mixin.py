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
  accepts by two properties of the shape the loop is rendered into.

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

    @classmethod
    def _schedule(cls, kernel):
        """Return the PSyIR schedule of the kernel to be captured.

        A kind-polymorphic kernel resolves to one schedule per specific
        procedure of its generic interface, and the one to capture is the one
        Fortran would have resolved the call to. That question is
        :py:meth:`~psyclone.domain.lfric.LFRicKern.validate_kernel_code_args`'s
        rather than this transformation's; see :py:meth:`_matches`.

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
        expansion = Reference2ArrayRangeTrans()
        lowering = ArrayAssignment2LoopsTrans()
        for assignment in schedule.walk(Assignment):
            if not cls._is_array_valued(assignment):
                continue
            if not isinstance(assignment.lhs, ArrayMixin):
                expansion.apply(assignment.lhs)
            lowering.apply(assignment)

    @staticmethod
    def _parallel_loops(schedule):
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

        Two rules narrow what it accepts, both of them properties of the shape
        the loop is rendered into rather than of the dependence analysis:

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

        :param schedule: the kernel schedule being captured, already lowered
            and bound-substituted, since both create loops.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: the loops to spread, outermost first, in schedule order.
        :rtype: Tuple[:py:class:`psyclone.psyir.nodes.Loop`, ...]
        """
        tools = DependencyTools()
        chosen = []
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
            if tools.can_loop_be_parallelised(loop):
                chosen.append(loop)
        return tuple(chosen)


__all__ = ["LFRicKokkosScheduleMixin"]
