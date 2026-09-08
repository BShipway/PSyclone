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

"""Say where a captured loop iterates, and what its launch is bounded by.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

It is a module of its own because the capture contract asks two unrelated
questions of a loop, and this is the first of them: *where the iterations
are*. Which of the loops a colouring leaves behind this is, which LFRic
iteration space it runs over, and between which two bounds -- and, once that
is settled, what the generated launch calls the count it runs to and the cell
it begins at. What those iterations may *write*, and how a write two cells
share is made safe, is ``LFRicKokkosWriteMixin``. The rest of the contract,
and the walk over the kernel body, is ``LFRicKokkosContractMixin``.

The launch's own bound names are here rather than in
``LFRicKokkosArgumentMixin`` because the iteration space decides them and
nothing else does:
:py:meth:`LFRicKokkosIterationMixin._count_name` reads the same space
:py:meth:`LFRicKokkosIterationMixin._validate_iteration_space` admitted, and
:py:meth:`LFRicKokkosIterationMixin._start_name` the same lower bound
:py:meth:`LFRicKokkosIterationMixin._validate_halo_depth` did. The argument
mixin asks for those names; it does not choose them.

The predicates are askable one at a time, because the coverage survey reports
every blocker a loop carries rather than the first:
:py:meth:`LFRicKokkosIterationMixin._validate_loop` calls
:py:meth:`LFRicKokkosIterationMixin._validate_builtin`,
:py:meth:`LFRicKokkosIterationMixin._validate_iteration_space` and
:py:meth:`LFRicKokkosIterationMixin._validate_halo_depth` rather than
repeating them. Each either returns or raises
:py:class:`~psyclone.psyir.transformations.TransformationError` naming what it
refused, and none of them alters the schedule it is given.

Nothing here reaches a sibling mixin: every name these methods use is on this
class. They are still resolved through ``cls`` on ``LFRicKokkosTrans``, which
is what lets a sibling ask for a launch bound without importing this module.
"""

from psyclone.domain.lfric import LFRicConstants
from psyclone.psyGen import BuiltIn
from psyclone.psyir.transformations import TransformationError


class LFRicKokkosIterationMixin:
    """Where a captured loop iterates, and what its launch is bounded by.

    Every question here is about the loop rather than about its kernel: which
    of the loops a colouring leaves this is, which iteration space it runs
    over, where it begins and what it counts to. The rest of the capture
    contract is ``LFRicKokkosContractMixin``; what those iterations may write
    is ``LFRicKokkosWriteMixin``; how the count and the first cell are then
    declared and passed is ``LFRicKokkosArgumentMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The one :py:attr:`~psyclone.psyGen.Loop.loop_type` a captured loop may
    #: carry besides none at all and ``dof``. It is the inner loop of a
    #: colouring, whose iterations are the cells of one colour; the enclosing
    #: ``colour`` loop is left as Fortran and is what runs the colours in
    #: sequence, which is where the safety of a shared write without atomics
    #: comes from.
    _COLOURED_LOOP_TYPE = "cells_in_colour"

    #: Loop types the launch has a shape for. ``""`` and ``None`` are a loop
    #: over cell columns, ``"dof"`` one over dofs, and
    #: :py:attr:`_COLOURED_LOOP_TYPE` the inner loop of a colouring, whose
    #: cells are read through a colour map rather than counted from the mesh.
    #: The enclosing ``colours`` loop and the tiled types are absent -- the
    #: first is what the PSy layer keeps and runs in sequence, and a tiled
    #: colouring has a second level of indirection nothing here models -- and
    #: ``"null"`` because it is not a loop at all: the kernel under it is
    #: called once, for the whole domain.
    _LOOP_TYPES = ("", None, "dof", _COLOURED_LOOP_TYPE)

    #: Iteration spaces the launch has a shape for. The four cell-column
    #: spaces all launch over cells, differing only in how far the count
    #: they are given reaches and, for ``halo_cell_column``, in where the
    #: launch begins; the two dof spaces launch over dofs. Listed rather
    #: than derived from
    #: :py:class:`~psyclone.domain.lfric.LFRicConstants`, so that a space
    #: LFRic adds later is refused by name instead of being accepted on the
    #: strength of resembling one of these.
    #:
    #: ``domain`` is the one LFRic space absent. A kernel operating on the
    #: whole domain is called once, with no loop for a launch to become.
    _ITERATION_SPACES = (
        "cell_column", "owned_cell_column", "halo_cell_column",
        "owned_and_halo_cell_column", "dof", "owned_dof")

    #: Lower bounds the launch can begin at. ``start`` is the first cell or
    #: dof, which the launch reaches by beginning at zero; ``cell_halo_start``
    #: is the first halo cell, which it reaches by taking
    #: :py:attr:`_CELL_START` as a formal of its own.
    #:
    #: The rest -- ``inner``, ``ncells`` and ``cell_halo`` -- are the lower
    #: bounds redundant computation produces, each of them relative to a
    #: depth index the region has no formal for. They are refused by name.
    _LOWER_BOUNDS = ("start", "cell_halo_start")

    #: Upper bounds a launch beginning at the first cell can cover. Each of
    #: these names a count of consecutive cells or dofs starting from the
    #: first, so the launch runs ``0`` to that count and every per-cell View
    #: is sliced to it: ``ncells`` the owned cells, ``cell_halo`` those and
    #: the halo to the depth the loop asks for, ``ndofs`` the owned dofs,
    #: ``nannexed`` those and the annexed ones, ``dof_halo`` the dofs to a
    #: halo depth. What each renders as in the PSy layer is
    #: :py:meth:`~psyclone.domain.lfric.LFRicLoop.upper_bound_psyir`'s
    #: business, and the region takes its value rather than its expression.
    #:
    #: The last two are the coloured pair, and they count the same way the
    #: others do: ``ncolour`` is the number of cells of the colour the
    #: enclosing loop is on, ``colour_halo`` that number to a halo depth.
    #: What differs is what the count is of -- cells of one colour rather
    #: than cells of the mesh -- and the region reads its cells through
    #: :py:class:`~psyclone.psyir.backend.kokkos.KokkosColourMap` for that
    #: reason. The tiled bounds, ``ntilecolours`` and the rest, remain
    #: absent: a tiled loop is refused by
    #: :py:meth:`_validate_iteration_space` before this is asked.
    _COUNTED_BOUNDS = ("ncells", "cell_halo", "ndofs", "nannexed", "dof_halo",
                       "ncolour", "colour_halo")

    #: The region's iteration count, and the second extent of every per-cell
    #: array, where the loop it came from iterated over cell columns. Named
    #: by the PSy layer, not by the kernel.
    _CELL_COUNT = "ncells"

    #: The same count where the loop iterated over dofs. A separate name
    #: rather than ``ncells`` reused, because the generated source is read:
    #: a region whose ``RangePolicy`` runs to ``ncells`` while its index is a
    #: dof would be telling a reviewer something untrue about what it does.
    _DOF_COUNT = "ndofs"

    #: The first cell of a launch that does not begin at the first cell of
    #: the mesh. Only a loop over the halo cells alone has one; see
    #: :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_start`.
    _CELL_START = "first_cell"

    #: The name a dof launch gives its own index, as
    #: :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_index`
    #: names a cell launch's. ``df`` is what every LFRic kernel calls the
    #: same thing.
    _DOF_INDEX = "df"

    #: Every name the generated signature may add for its own bounds. All
    #: three are reserved for every region, whichever of them that region
    #: goes on to use, so that whether a kernel is refused for a name
    #: collision does not depend on which iteration space its loop had.
    _BOUND_NAMES = (_CELL_COUNT, _DOF_COUNT, _CELL_START)

    @classmethod
    def _validate_iteration_space(cls, node):
        """Check that the launch has a shape for what the loop iterates over.

        Two questions, asked separately because they have separate answers.
        The *type* says which of the loops a colouring leaves behind this is,
        and only the inner one is captured: it runs the cells of one colour,
        which the region reads through a
        :py:class:`~psyclone.psyir.backend.kokkos.KokkosColourMap`, while the
        enclosing ``colours`` loop stays in the PSy layer and is what runs the
        colours one after another. That sequence is the alternative to an
        atomic update, and it is why a coloured loop is admitted at all. Every
        other type -- the enclosing loop, a tiled colouring's -- is refused.
        The space is then refused by name, because the survey reports one
        blocker per loop and the name is what tells two unadmitted spaces
        apart.

        Both the cell-column spaces and the dof spaces are admitted. A dof
        loop hands its kernel one dof of each field rather than a dofmap, so
        it needs no cell index and no indirection at all; see
        :py:func:`~psyclone.psyir.backend.kokkos_launch_dof.dof_launch`. A
        colouring never reaches a dof loop --
        :py:class:`~psyclone.transformations.LFRicColourTrans` refuses one --
        so the two spaces and the coloured type are checked independently
        here and can be relied on not to arrive together.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :raises TransformationError: if the loop's type is not one of
            :py:attr:`_LOOP_TYPES`.
        :raises TransformationError: if the loop's iteration space is not one
            of :py:attr:`_ITERATION_SPACES`.
        """
        if node.loop_type not in cls._LOOP_TYPES:
            raise TransformationError(
                "LFRicKokkosTrans supports only a loop over cell columns, "
                "coloured or not, or a loop over dofs.")
        if node.iteration_space not in cls._ITERATION_SPACES:
            raise TransformationError(
                f"LFRicKokkosTrans does not support the "
                f"'{node.iteration_space}' iteration space.")

    @classmethod
    def _validate_halo_depth(cls, node):
        """Check that the loop runs from the first cell or dof to a count.

        The depth and the bound names are one question rather than two. A
        loop written over the halo carries a depth; one written over
        ``cell_halo`` with no depth carries none and is the same fact stated
        differently, so a survey that reported them apart would count one
        blocked pattern twice.

        The upper bound is not the question it once was. A region is launched
        over whatever count its loop carried -- the count crosses the ABI as
        :py:attr:`_CELL_COUNT`, filled from the loop's own stop expression
        -- so a bound reaching into the halo needs
        nothing of the generated code that the owned-cell bound did not. What
        it needs of the *bound* is that it be a count from the first cell or
        dof, which is what :py:attr:`_COUNTED_BOUNDS` lists.

        The lower bound is now a question of the same shape. A loop over the
        halo cells alone begins where the owned cells end, and the region
        carries that first cell as a formal beside the count, exactly as it
        carries the count: the PSy layer holds the value and the launch takes
        it. What is refused is a lower bound the PSy layer states relative to
        a depth index the region has no formal for, which is what
        :py:attr:`_LOWER_BOUNDS` excludes. Refusing rather than trusting the
        upper bound alone still matters: a launch beginning at zero over a
        loop that did not would run the cells it was told to skip, which is a
        wrong answer rather than a compile error.

        **A coloured loop always begins at ``start``**, so a coloured region
        never takes a first cell, and the writer refuses one that does. That
        is not this rule's doing but
        :py:class:`~psyclone.transformations.LFRicColourTrans`'s: the loop it
        makes over the cells of one colour is given a lower bound of
        ``start`` whatever the loop it replaced had, the halo the original
        reached into moving into the *upper* bound, ``colour_halo``. The two
        counts a launch could otherwise be given -- an index into one colour's
        cells and a cell of the mesh -- are therefore never asked to compose.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :raises TransformationError: if the loop's lower bound is not one of
            :py:attr:`_LOWER_BOUNDS`.
        :raises TransformationError: if the loop's upper bound is not one of
            :py:attr:`_COUNTED_BOUNDS`.
        """
        # LFRicLoop does not currently expose its lower-bound name.
        # pylint: disable=protected-access
        if node._lower_bound_name not in cls._LOWER_BOUNDS:
            raise TransformationError(
                f"LFRicKokkosTrans does not support the "
                f"'{node._lower_bound_name}' loop lower bound.")
        if node.upper_bound_name not in cls._COUNTED_BOUNDS:
            raise TransformationError(
                f"LFRicKokkosTrans does not support the "
                f"'{node.upper_bound_name}' loop bound.")

    @staticmethod
    def _validate_builtin(node):
        """Refuse a loop holding an LFRic builtin, by the builtin's name.

        LFRic writes a builtin as a loop over dofs, so admitting the dof
        iteration space reaches them and this rule is what stops it. A
        builtin has no kernel file and no
        :py:class:`~psyclone.psyir.nodes.KernelSchedule`: PSyclone lowers it
        into the PSy layer itself. Every rule after this one asks a question
        of that schedule, so without this the capture fails with an
        ``AttributeError`` from inside the metadata checks rather than with a
        refusal -- which the coverage survey would record as an error row and
        a whole-model capture build would stop on.

        Capturing a builtin is a capability of its own: what would be
        generated is not a translation of a kernel file but of PSyclone's own
        model of the operation.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :raises TransformationError: if any kernel in the loop is a builtin.
        """
        for kernel in node.kernels():
            if isinstance(kernel, BuiltIn):
                raise TransformationError(
                    f"LFRicKokkosTrans does not support the LFRic builtin "
                    f"'{kernel.name}'.")

    @classmethod
    def _validate_dof_body(cls, node, schedule, parallel_loops):
        """Refuse a dof kernel asking for anything only a team can give.

        The dof launch is a flat range and has no team: no per-thread scratch
        to place a kernel-local array in, and no members to spread a loop
        across. Both of those are things the two cell launches offer, and
        both are chosen by looking at the kernel rather than at the loop, so
        a dof kernel carrying one would otherwise reach
        :py:class:`~psyclone.psyir.backend.kokkos.KokkosWriter` describing a
        region it refuses -- a failure part-way through the capture rather
        than a refusal, and a loop the coverage survey had called capturable.

        Neither is a shape a GungHo dof kernel has: a kernel handed one dof
        of each field has nothing to size a local array by. It is refused
        rather than left unexamined because the survey is asked of every loop
        in the model and reports what it is told.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param schedule: the kernel schedule being captured, carrying every
            rewrite ``apply`` makes; the same probe
            ``LFRicKokkosContractMixin._validate_locals`` is asked of,
            because an allocated local only has its shape once the
            allocation tier is lowered.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param parallel_loops: the loops the launch would spread over a team,
            as ``_parallel_loops`` gives them.
        :type parallel_loops: Tuple[:py:class:`psyclone.psyir.nodes.Loop`,
            ...]

        :raises TransformationError: if the loop is over dofs and its kernel
            declares an automatic array.
        :raises TransformationError: if the loop is over dofs and its kernel
            has a loop that would be spread over a team.
        """
        if not cls._is_dof(node):
            return
        # Sorted so that a kernel with two of them names the same one on
        # every run.
        for symbol in sorted(schedule.symbol_table.automatic_datasymbols,
                             key=lambda symbol: symbol.name):
            if symbol.is_array:
                raise TransformationError(
                    f"LFRicKokkosTrans cannot place the kernel-local array "
                    f"'{symbol.name}' of a loop over dofs: the dof launch is "
                    "a flat range and has no team to hold scratch.")
        if parallel_loops:
            raise TransformationError(
                "LFRicKokkosTrans cannot spread a loop of a kernel over dofs "
                "across a team: the dof launch is a flat range and has no "
                "team.")

    @classmethod
    def _validate_loop(cls, node):
        """Check the loop's own iteration contract.

        The two rules about where the loop iterates are
        :py:meth:`_validate_iteration_space` and
        :py:meth:`_validate_halo_depth`, called here in the order they have
        always been checked so that a loop failing more than one of them
        reports the same refusal as before.
        :py:meth:`_validate_builtin` precedes both, because a builtin has no
        kernel schedule for the rules after it to be asked of.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :raises TransformationError: if the loop does not hold exactly one
            kernel.
        """
        cls._validate_builtin(node)
        cls._validate_iteration_space(node)
        cls._validate_halo_depth(node)
        if len(node.kernels()) != 1:
            raise TransformationError(
                "LFRicKokkosTrans requires exactly one kernel in the loop.")

    @classmethod
    def _is_dof(cls, node):
        """Say whether ``node`` iterates over dofs rather than cell columns.

        :param node: the loop being captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :returns: whether the loop's iteration space is a dof one.
        :rtype: bool
        """
        return node.iteration_space in LFRicConstants().DOF_ITERATION_SPACES

    @classmethod
    def _count_name(cls, node):
        """Name the formal the launch for ``node`` is bounded above by.

        :param node: the loop being captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :returns: :py:attr:`_DOF_COUNT` for a loop over dofs and
            :py:attr:`_CELL_COUNT` for one over cell columns.
        :rtype: str
        """
        return cls._DOF_COUNT if cls._is_dof(node) else cls._CELL_COUNT

    @classmethod
    def _start_name(cls, node):
        """Name the formal the launch for ``node`` begins at, or ``None``.

        A loop starting anywhere but at the first cell or dof needs one; that
        is the halo-only iteration space and nothing else, because every
        other bound this transformation accepts counts from the first.

        :param node: the loop being captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :returns: :py:attr:`_CELL_START` where the loop begins past the first
            cell, and ``None`` where it does not.
        :rtype: Optional[str]
        """
        # pylint: disable-next=protected-access
        return None if node._lower_bound_name == "start" else cls._CELL_START


__all__ = ["LFRicKokkosIterationMixin"]
