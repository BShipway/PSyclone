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

"""Decide whether a loop may be captured, and refuse it by name if not.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

Every method here is a predicate that either returns or raises
:py:class:`~psyclone.psyir.transformations.TransformationError` naming what it
refused and why, and none of them alters the schedule it is given: where a
prediction needs a rewrite to have happened, it is made over a copy. That is
what lets ``LFRicKokkosTrans.validate`` be called for its answer alone -- by
the coverage survey, which asks it of every loop in the model and keeps the
message.

The survey reports every blocker a loop carries rather than the first, so the
rules ``_validate_loop`` and ``_validate_kernel_metadata`` bundle are each
askable on their own:
:py:meth:`LFRicKokkosContractMixin._validate_iteration_space`,
:py:meth:`LFRicKokkosContractMixin._validate_halo_depth`,
:py:meth:`LFRicKokkosContractMixin._validate_evaluator`,
:py:meth:`LFRicKokkosContractMixin._validate_intergrid`,
:py:meth:`LFRicKokkosContractMixin._validate_field_types` and
:py:meth:`LFRicKokkosContractMixin._validate_continuous_write`. The two
bundling methods call them rather than repeating them, and the last two share
the per-argument
:py:meth:`LFRicKokkosContractMixin._validate_field_type` and
:py:meth:`LFRicKokkosContractMixin._validate_written_space` with the argument
walk in ``_validate_kernel_metadata``, so the order the bundled refusals come
in is unchanged by their being nameable apart.

The sibling mixins are reached through ``cls``, resolved on
``LFRicKokkosTrans``: :py:meth:`LFRicKokkosContractMixin._validate_sections`
and :py:meth:`LFRicKokkosContractMixin._validate_bounds` predict
``cls._lower_sections`` and ``cls._substitute_bounds`` over copies, and
:py:meth:`LFRicKokkosContractMixin._validate_formals` and
:py:meth:`LFRicKokkosContractMixin._validate_locals` ask ``cls._c_type``,
``cls._extents`` and ``cls._CELL_COUNT``. Calling a method here directly on
this mixin is therefore not supported.
"""

from psyclone.core import AccessType
from psyclone.domain.lfric import LFRicConstants
from psyclone.psyir.nodes import (
    Assignment, Call, CodeBlock, IntrinsicCall, Range, Reference)
from psyclone.psyir.transformations import TransformationError


class LFRicKokkosContractMixin:
    """The capture contract, as a set of side-effect-free predicates.

    Every question here is whether a loop, a kernel or a schedule is inside
    what the generated region can express. What a symbol is in C terms is
    ``LFRicKokkosTypesMixin``; how the region and its call are built is
    ``LFRicKokkosCallMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: Accesses a plain ``parallel_for`` over cells can honour. ``INC``,
    #: ``READINC`` and ``REDUCTION`` all need colouring or atomics.
    _SAFE_ACCESSES = (AccessType.READ, AccessType.WRITE, AccessType.READWRITE)
    #: The LFRic argument types the region can describe. ``gh_operator`` is an
    #: LMA operator, which reaches the kernel as a rank-3 array over
    #: ``(ncell_3d, ndf1, ndf2)`` with every extent a formal of its own, so
    #: the existing View description covers it whole.
    #: ``gh_columnwise_operator`` is absent and is the CMA case: a banded
    #: matrix carrying its own bandwidth and indexing arguments, none of
    #: which this region has.
    _SUPPORTED_ARGUMENTS = ("gh_field", "gh_scalar", "gh_operator")

    #: Identifiers the generated launch declares in the scope the kernel
    #: body is generated into. A kernel-local of the same name would
    #: shadow the launch's own and then overwrite it, which is a wrong
    #: answer rather than a compile error. The cell index is absent
    #: because it is renamed instead; see
    #: :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_index`.
    #: These seven belong to the two team launches, so they are checked for a
    #: kernel that selects one: an automatic array selects the flat team
    #: launch and a parallelisable level loop the hierarchical one, and the
    #: hierarchical launch declares ``team`` whether or not there is scratch.
    #: :py:attr:`_CELL_COUNT` is declared by all three shapes and is always
    #: checked.
    _GENERATED_NAMES = ("body", "league_size", "probe", "rank",
                        "scratch_bytes", "team", "team_size")

    #: Stencil shapes whose PSy-layer arguments :py:meth:`apply`'s generic
    #: per-cell rule already passes correctly. A 2-D stencil hands the kernel
    #: a sliced dofmap and a sliced size array, both of them array formals, so
    #: both become Views with the cell index appended -- which is what
    #: ``map_w3(:,cell)`` already does and needs nothing new.
    #:
    #: A 1-D or region stencil hands the size to the kernel as a *scalar*
    #: formal, fed from ``x_stencil_size(cell)``. That rule would pass the
    #: whole sliced expression against a by-value dummy, so admitting those
    #: shapes needs a per-cell scalar argument kind that does not exist here.
    #: No executed GungHo loop asks for one, so they are refused by name.
    _SUPPORTED_STENCILS = ("cross2d",)

    @staticmethod
    def _validate_iteration_space(node):
        """Check that the loop iterates over uncoloured cell columns.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :raises TransformationError: if the loop is coloured or is not over
            cell columns.
        """
        if node.loop_type or node.iteration_space != "cell_column":
            raise TransformationError(
                "LFRicKokkosTrans supports only an uncoloured cell-column "
                "loop.")

    @staticmethod
    def _validate_halo_depth(node):
        """Check that the loop visits the owned cells and no others.

        The depth and the bound names are one question rather than two. A
        loop written over the halo carries a depth; one written over
        ``cell_halo`` with no depth carries none and is the same fact stated
        differently, so a survey that reported them apart would count one
        blocked pattern twice.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :raises TransformationError: if the loop carries a halo depth.
        :raises TransformationError: if the loop is not bounded by the owned
            cells.
        """
        if node.upper_bound_halo_depth is not None:
            raise TransformationError(
                "LFRicKokkosTrans does not support a halo depth.")
        # LFRicLoop does not currently expose its lower-bound name.
        # pylint: disable=protected-access
        if (node._lower_bound_name != "start" or
                node.upper_bound_name != "ncells"):
            raise TransformationError(
                "LFRicKokkosTrans supports only owned-cell bounds.")

    @classmethod
    def _validate_loop(cls, node):
        """Check the loop's own iteration contract.

        The two rules about where the loop iterates are
        :py:meth:`_validate_iteration_space` and
        :py:meth:`_validate_halo_depth`, called here in the order they have
        always been checked so that a loop failing more than one of them
        reports the same refusal as before.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :raises TransformationError: if the loop does not hold exactly one
            kernel.
        """
        cls._validate_iteration_space(node)
        cls._validate_halo_depth(node)
        if len(node.kernels()) != 1:
            raise TransformationError(
                "LFRicKokkosTrans requires exactly one kernel in the loop.")

    @staticmethod
    def _validate_evaluator(kernel):
        """Check that the kernel asks for no quadrature or evaluator data.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :raises TransformationError: if the kernel needs quadrature or
            evaluator data.
        """
        if kernel.qr_required or kernel.eval_shapes:
            raise TransformationError(
                "LFRicKokkosTrans does not support quadrature or evaluator "
                "data.")

    @staticmethod
    def _validate_intergrid(kernel):
        """Check that the kernel iterates over a single mesh.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :raises TransformationError: if the kernel is an inter-grid kernel.
        """
        if kernel.is_intergrid:
            raise TransformationError(
                "LFRicKokkosTrans does not support inter-grid kernels.")

    @staticmethod
    def _validate_field_type(argument):
        """Check one argument's intrinsic type, if it is a field.

        Anything that is not a field is passed over rather than refused, so
        that the rule can be walked over a whole argument list.

        :param argument: the kernel argument to check.
        :type argument: :py:class:`psyclone.lfric.LFRicKernelArgument`

        :raises TransformationError: if the argument is a field whose data is
            not real.
        """
        if argument.argument_type != "gh_field":
            return
        if argument.intrinsic_type != "real":
            raise TransformationError(
                f"LFRicKokkosTrans supports only real fields, but "
                f"'{argument.name}' is {argument.intrinsic_type}.")

    @staticmethod
    def _validate_written_space(argument, discontinuous):
        """Check one argument's function space, if it is a written field.

        A read field is passed over, as is anything that is not a field:
        only a written space decides whether cells may run in parallel.

        :param argument: the kernel argument to check.
        :type argument: :py:class:`psyclone.lfric.LFRicKernelArgument`
        :param discontinuous: the names of the discontinuous function spaces,
            as :py:class:`psyclone.domain.lfric.LFRicConstants` gives them.
        :type discontinuous: List[str]

        :raises TransformationError: if the argument is a field written on a
            continuous space, where one cell's contribution could overwrite
            another's.
        """
        if argument.argument_type != "gh_field":
            return
        if argument.access == AccessType.READ:
            return
        space = argument.function_space.orig_name.lower()
        if space not in discontinuous:
            raise TransformationError(
                f"LFRicKokkosTrans requires a discontinuous space for "
                f"the written field '{argument.name}', but found "
                f"'{space}': one cell's contribution could overwrite "
                "another's.")

    @classmethod
    def _validate_field_types(cls, kernel):
        """Check that every field the kernel takes is real.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :raises TransformationError: if any field argument's data is not
            real, for the reason :py:meth:`_validate_field_type` gives.
        """
        for argument in kernel.arguments.args:
            cls._validate_field_type(argument)

    @classmethod
    def _validate_continuous_write(cls, kernel):
        """Check every field the kernel writes for a discontinuous space.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :raises TransformationError: if any written field argument is on a
            continuous space, for the reason
            :py:meth:`_validate_written_space` gives.
        """
        discontinuous = LFRicConstants().VALID_DISCONTINUOUS_NAMES
        for argument in kernel.arguments.args:
            cls._validate_written_space(argument, discontinuous)

    @classmethod
    def _validate_kernel_metadata(cls, kernel):
        """Check the LFRic metadata of the kernel to be captured.

        The order the refusals come in is part of the contract: the checks
        with no argument of their own run first, and then the argument list
        is walked once, each argument answering every rule before the next
        argument is looked at. A kernel with two blockers on two arguments
        therefore reports the first *argument's*, which is not what running
        :py:meth:`_validate_field_types` and
        :py:meth:`_validate_continuous_write` in turn would report. Those two
        share :py:meth:`_validate_field_type` and
        :py:meth:`_validate_written_space` with the walk below rather than
        restating them, so there is one copy of each rule and two ways to ask
        it.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :raises TransformationError: if the kernel needs quadrature or
            evaluator data, is a CMA or inter-grid kernel, takes an argument
            that is not a field, a scalar or an LMA operator, takes an access
            a cell-parallel launch cannot honour, takes a non-real field, uses
            a stencil shape outside :py:attr:`_SUPPORTED_STENCILS`, or writes
            to a field on a continuous space.
        """
        cls._validate_evaluator(kernel)
        if kernel.cma_operation is not None:
            raise TransformationError(
                "LFRicKokkosTrans does not support CMA operators.")
        cls._validate_intergrid(kernel)

        discontinuous = LFRicConstants().VALID_DISCONTINUOUS_NAMES
        for argument in kernel.arguments.args:
            # An LMA operator needs no rule of its own beyond this one. It
            # reaches the kernel as a rank-3 array whose every extent is
            # itself a formal, so the existing View description covers it, and
            # a cell's slice of it is disjoint from every other cell's by
            # construction -- so a cell-parallel launch cannot race on it
            # whatever space it is built over.
            if argument.argument_type not in cls._SUPPORTED_ARGUMENTS:
                raise TransformationError(
                    "LFRicKokkosTrans supports only "
                    f"{', '.join(cls._SUPPORTED_ARGUMENTS)} arguments, "
                    f"found '{argument.argument_type}'.")
            if argument.access not in cls._SAFE_ACCESSES:
                raise TransformationError(
                    f"LFRicKokkosTrans cannot capture the '{argument.access}' "
                    f"access of '{argument.name}': a cell-parallel launch "
                    "would need colouring or atomics.")
            if argument.argument_type != "gh_field":
                continue
            cls._validate_field_type(argument)
            if argument.stencil:
                shape = str(argument.stencil.name).lower()
                if shape not in cls._SUPPORTED_STENCILS:
                    raise TransformationError(
                        f"LFRicKokkosTrans supports the "
                        f"{', '.join(cls._SUPPORTED_STENCILS)} stencil shape "
                        f"only, but '{argument.name}' has '{shape}'.")
            cls._validate_written_space(argument, discontinuous)

    @staticmethod
    def _validate_body(schedule):
        """Check that nothing in the body escapes the generated region.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if the body holds a
            :py:class:`~psyclone.psyir.nodes.CodeBlock`, or a call to
            anything other than an intrinsic, neither of which the generated
            region has any way to express.
        """
        if schedule.walk(CodeBlock):
            raise TransformationError(
                "LFRicKokkosTrans cannot capture a CodeBlock.")
        for call in schedule.walk(Call):
            if isinstance(call, IntrinsicCall):
                continue
            routine = call.routine
            name = routine.symbol.name if routine else "an unnamed routine"
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture the call to '{name}': the "
                "generated region has no Fortran to call into.")

    @classmethod
    def _validate_sections(cls, schedule):
        """Check that every whole-column section can be lowered to a loop.

        The generated region has no way to say ``a(i:j)``, so a section is
        rewritten as an explicit loop before the backend sees it. This
        predicts that rewrite rather than performing it, because
        :py:meth:`validate` must leave the schedule as it found it.

        The prediction is made by lowering a **copy** of the schedule rather
        than by asking ``ArrayAssignment2LoopsTrans.validate``, whose answer is
        weaker than its own ``apply``: it skips the references a ``Call``
        encloses, and ``apply`` expands them like any other, so an unresolved
        import used as an intrinsic's argument is accepted and then refused.
        Restating that rule here would leave two copies of it to keep in step,
        so the lowering itself is the predicate and the copy is what keeps it
        side-effect free.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if a section is not part of an
            assignment, and so is beyond what lowering can reach.
        :raises TransformationError: if a section's own assignment cannot be
            lowered, in which case the reason is the one PSyclone gives.
        """
        for section in schedule.walk(Range):
            if section.ancestor(Assignment) is None:
                raise TransformationError(
                    "LFRicKokkosTrans cannot capture an array section outside "
                    "an assignment: only a whole-column assignment can be "
                    "lowered to a loop the generated region can express.")
        if not schedule.walk(Range):
            return
        try:
            cls._lower_sections(schedule.copy())
        except TransformationError as err:
            raise TransformationError(
                "LFRicKokkosTrans cannot lower an array section to a "
                f"loop: {err}") from err

    @classmethod
    def _validate_bounds(cls, schedule):
        """Check that every shape enquiry resolves to a declared bound.

        Predicts :py:meth:`_substitute_bounds` over a copy, as
        :py:meth:`_validate_sections` predicts the lowering, because that
        substitution mutates the schedule and :py:meth:`validate` must leave
        it as it found it.

        The copy is lowered first. ``ArrayAssignment2LoopsTrans`` is a
        *producer* of ``LBOUND`` and ``UBOUND``, writing them into the loop
        bounds of every full-extent section it rewrites, so checking before
        the lowering would miss the calls the transformation itself creates.
        On a schedule that is already lowered -- which is what the coverage
        survey hands this method -- the lowering is a no-op, so the one
        method serves both callers. The cost is a second schedule copy per
        validation, taken so that each predicate stays readable alone.

        A schedule whose sections cannot be lowered says nothing about its
        bounds that :py:meth:`_validate_sections` has not already said, so
        that failure is passed over rather than re-reported. Reaching it means
        this method was called on its own, as the coverage survey calls each
        predicate independently; from :py:meth:`validate` the section check
        has refused the schedule before this runs.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if a shape enquiry cannot be resolved
            from the declaration, for any of the reasons
            :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosTypesMixin._substitute_bounds` gives.
        """
        probe = schedule.copy()
        try:
            cls._lower_sections(probe)
        except TransformationError:
            return
        try:
            cls._substitute_bounds(probe)
        except TransformationError as err:
            raise TransformationError(
                "LFRicKokkosTrans cannot resolve an array bound from its "
                f"declaration: {err}") from err

    @classmethod
    def _validate_formals(cls, schedule):
        """Check that every kernel formal has a place on the C ABI.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if the kernel already declares the name
            the generated signature adds for the cell count.
        :raises TransformationError: if a formal's kind is not one
            :py:attr:`_C_TYPES` maps, or if an array formal's extent is not
            itself a formal, so the generated View could not be sized.
        """
        table = schedule.symbol_table
        formals = table.argument_list
        names = {symbol.name for symbol in formals}
        if cls._CELL_COUNT in names:
            raise TransformationError(
                f"LFRicKokkosTrans adds '{cls._CELL_COUNT}' to the generated "
                "signature, but the kernel already declares it.")
        for symbol in formals:
            if cls._c_type(symbol) is None:
                raise TransformationError(
                    f"LFRicKokkosTrans supports {cls._supported_kinds()} "
                    f"argument kinds only, but '{symbol.name}' has "
                    f"'{cls._kind_name(symbol)}'.")
            # Sorted because the refusal names one offender and a set does
            # not iterate in a fixed order, so an unsorted loop would give a
            # different message run to run for the same kernel.
            for name in sorted(cls._extent_names(symbol)):
                if name not in names:
                    raise TransformationError(
                        f"LFRicKokkosTrans needs the extent '{name}' of "
                        f"'{symbol.name}' to be a kernel argument, so that "
                        "the generated View can be sized.")

    @classmethod
    def _validate_locals(cls, schedule, parallel_loops=()):
        """Check that every kernel-local array can be placed in team scratch.

        An automatic array is a per-cell temporary whose extent is a runtime
        value, so the region places it in scratch rather than declaring it in
        the body -- per team rank under the flat launch, where a rank is a
        cell, and per team under the hierarchical one, where a team is.
        Sizing that View is what the two refusals below protect: the element
        type has to be one the ABI names, and the extent has to be a value
        the launch already holds, which is a kernel argument.

        A module constant is refused as an extent even though
        :py:meth:`_constants` could import one, because the launch computes
        its scratch size before it enters the region.

        A local is also refused for its *name* alone, where that name is one
        the generated launch declares in the scope the kernel body is
        generated into. The kernel's declaration would shadow the launch's
        and then be assigned to, so the region would run with a value the
        kernel had overwritten -- a wrong answer, where the launch index's
        collision is a compile error. The launch index is renamed around that
        collision rather than refused because it is one name in two places;
        these are threaded through both launch shapes and the scratch sizing,
        and no GungHo kernel declares one.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param parallel_loops: the loops the launch will spread over the team,
            as :py:meth:`_parallel_loops` gives them. Only whether there are
            any is read: a non-empty tuple selects the hierarchical launch,
            which declares ``team`` with or without scratch, so the names
            below are reserved for a kernel that has one even if it has no
            automatic array to place.
        :type parallel_loops: Tuple[:py:class:`psyclone.psyir.nodes.Loop`, ...]

        :raises TransformationError: if the kernel declares a local named
            :py:attr:`_CELL_COUNT`, which all three launch shapes declare.
        :raises TransformationError: if the kernel selects a team launch --
            by having an automatic array, or a loop to spread over the team --
            and declares a local named in :py:attr:`_GENERATED_NAMES`, which
            that launch declares.
        :raises TransformationError: if a local array's kind is not one
            :py:attr:`_C_TYPES` maps, or if one of its extents is not a
            kernel argument, so the scratch View could not be sized.
        """
        table = schedule.symbol_table
        names = {symbol.name for symbol in table.argument_list}

        locals_ = list(table.automatic_datasymbols)
        generated = {cls._CELL_COUNT}
        if any(symbol.is_array for symbol in locals_) or bool(parallel_loops):
            generated.update(cls._GENERATED_NAMES)
        # Sorted so that a kernel colliding with two of them names the same
        # one on every run.
        for name in sorted(generated.intersection(
                symbol.name for symbol in locals_)):
            raise TransformationError(
                f"LFRicKokkosTrans' generated launch declares '{name}', but "
                "the kernel declares a local of that name.")

        for symbol in table.automatic_datasymbols:
            if not symbol.is_array:
                continue
            if cls._c_type(symbol) is None:
                raise TransformationError(
                    f"LFRicKokkosTrans supports {cls._supported_kinds()} "
                    f"kernel-local array kinds only, but '{symbol.name}' has "
                    f"'{cls._kind_name(symbol)}'.")
            for name in sorted(cls._extent_names(symbol)):
                if name not in names:
                    raise TransformationError(
                        f"LFRicKokkosTrans needs the extent '{name}' of "
                        f"the kernel-local array '{symbol.name}' to be a "
                        "kernel argument, so that the generated scratch can "
                        "be sized.")

    @staticmethod
    def _cell_position(kernel, loop, formals, actuals):
        """Return the formal carrying LFRic's cell index, or ``None``.

        ``ArgOrdering.generate`` emits that index as the kernel's first
        argument for exactly the kernels that take an operator, so
        ``has_operator()`` is both the test and the position. The kernel uses
        the value arithmetically -- ``ik = (cell - 1) * nlayers + 1`` -- to
        find its own slice of an operator's local stencil, and the region
        declares it from the launch's own index rather than taking it, since
        the actual the PSy layer supplies is the loop counter that lowering
        removes.

        That the actual *is* the loop counter is checked rather than assumed.
        If ``ArgOrdering`` and this method ever part company, the entry
        dropped below is some other argument, and the result is a region that
        compiles, runs and reads the wrong data.

        :param kernel: the kernel being captured.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`
        :param loop: the loop the kernel sits in.
        :type loop: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param formals: the kernel's formal arguments, in order.
        :type formals: list[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        :param actuals: the actual arguments the PSy layer supplies, in the
            same order.
        :type actuals: list[:py:class:`psyclone.psyir.nodes.Node`]

        :returns: the name of the cell-position formal, or ``None``.
        :rtype: Optional[str]

        :raises TransformationError: if the kernel takes an operator but the
            first actual is not a reference to the loop's own variable.
        """
        if not kernel.arguments.has_operator():
            return None
        actual = actuals[0]
        if not (isinstance(actual, Reference)
                and actual.symbol is loop.variable):
            raise TransformationError(
                f"LFRicKokkosTrans expected the PSy layer to supply the "
                f"loop's own cell index as the first argument of "
                f"'{kernel.name}', but found '{actual.debug_string()}'.")
        return formals[0].name


__all__ = ["LFRicKokkosContractMixin"]
