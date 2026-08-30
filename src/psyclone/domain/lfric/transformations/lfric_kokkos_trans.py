# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Capture a supported LFRic loop as a Kokkos launch."""

from psyclone.core import AccessType
from psyclone.domain.lfric import KernCallArgList, LFRicConstants, LFRicLoop
from psyclone.domain.lfric.transformations.lfric_kokkos_call_mixin import (
    LFRicKokkosCallMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_types_mixin import (
    LFRicKokkosTypesMixin)
from psyclone.errors import GenerationError
from psyclone.psyGen import InvokeSchedule, Transformation
from psyclone.psyir.backend.kokkos import KokkosRegion, KokkosWriter
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import (
    ArrayReference, Assignment, Call, CodeBlock, IntrinsicCall, Range,
    Reference, Routine)
from psyclone.psyir.transformations import (
    ArrayAssignment2LoopsTrans, TransformationError)


class LFRicKokkosTrans(LFRicKokkosTypesMixin, LFRicKokkosCallMixin,
                       Transformation):
    """Replace one supported LFRic cell-column loop with a C ABI call.

    The transformation recognises a kernel shape rather than a named kernel:
    an uncoloured owned-cell loop over a single kernel whose arguments are
    fields and scalars, whose written fields are on discontinuous spaces, and
    whose formals and referenced module constants all map onto the
    ``int``/``float``/``double`` ABI the Kokkos backend emits. Every part of
    the generated region -- its name, its C signature, its Views and the
    ``bind(C)`` interface the PSy layer calls through -- is derived from that
    kernel, so a second kernel needs no change here.

    **Precision is carried, not chosen.** A kind is placed on the ABI by the
    width LFRic's precision map gives it, so an ``r_solver`` kernel reaches
    C++ as ``float`` in a single-precision build and as ``double`` in a
    double-precision one, from the same source and with no option to set. The
    region computes at that width throughout: its locals are declared at their
    own kind and its literals are suffixed, because neither crosses an
    interface and both would otherwise be generated at the C writer's default
    of ``double``, silently promoting the kernel. The generated ``bind(C)``
    interface then asserts, at compile time, that each kind really has the
    width the C++ was generated for, so a rebuild at another precision is a
    compile error naming the kind rather than a wrong answer.

    **A kind-polymorphic kernel is resolved, not refused.** LFRic writes such
    a kernel as a generic interface over specific procedures differing only in
    the precision of their real arguments, and PSyclone presents one schedule
    per procedure. The one captured is the one the algorithm layer's
    precisions select, which is
    :py:meth:`~psyclone.domain.lfric.LFRicKern.validate_kernel_code_args`'s
    question rather than this transformation's -- it exists to identify the
    right subroutine of a mixed-precision kernel, and matches in byte widths
    through the same precision map. The region takes the name of the selected
    procedure rather than of the interface, so two invokes of one kernel at
    different precisions generate two regions instead of colliding on one.

    Matching in widths is what makes the ABI right and the choice sometimes
    impossible. ``r_single`` and ``r_solver`` are both 4 bytes, so an
    interface offering both is refused rather than resolved by coincidence,
    as is one no algorithm precision selects at all. Metadata the matcher
    cannot model -- a stencil, an evaluator shape, a CMA or inter-grid kernel,
    which PSyclone's issue #928 leaves unbuilt -- is a third refusal, kept
    distinct from finding no match because a question that cannot be asked has
    not been answered "no".

    A kind the ABI does not name is refused rather than guessed at. That
    includes every ``logical`` kind: LFRic's ``l_def`` is ``kind(.false.)``,
    which is 4 bytes, so passing it as ``logical(c_bool)`` would put a 1-byte
    formal against a 4-byte actual. Admitting logicals needs that mismatch
    resolved, not a table entry.

    A whole-column array section such as ``a(i:j)``, which the finite-volume
    kernels use to assign a column as a unit, is accepted and lowered to an
    explicit loop by
    :py:class:`~psyclone.psyir.transformations.ArrayAssignment2LoopsTrans`
    before the region is described. The generated region has no way to say
    ``a(i:j)``, so a section that transformation refuses -- one carrying a
    loop-carried dependency, for instance -- is refused here too, with its
    reason quoted. A section outside an assignment altogether is beyond what
    lowering can reach and is refused before the backend sees it.

    ``LBOUND``, ``UBOUND`` and ``SIZE`` are resolved from the declaration
    rather than evaluated. Each is replaced by the bound the kernel's own
    symbol table gives, so ``UBOUND(partial, 1)`` on a local declared
    ``dimension(nlayers)`` becomes ``nlayers`` and the region never asks a
    View for a shape the Fortran has already stated. Most of them are not
    written by the kernel author at all: the section lowering above puts them
    into the loop bounds of every full-extent assignment it rewrites, which is
    why the substitution runs after that lowering rather than before it. What
    is required is a plain reference to a declared array and a dimension given
    as an integer literal within its rank -- ``SIZE(a)`` needs no dimension
    only when ``a`` is rank 1 -- and the extents themselves must satisfy the
    grammar described below, whose refusal is passed through unchanged.

    A kernel-local automatic array -- a temporary such as
    ``real(kind=r_def), dimension(nlayers) :: x_new``, whose extent is known
    only at runtime -- is placed in Kokkos team scratch, one private View per
    team rank, so that the cells sharing a team do not share a temporary. The
    region is then launched over a ``TeamPolicy`` rather than a
    ``RangePolicy``; a kernel with no array locals keeps the flat launch.

    That placement is what the refusals protect. The array's element kind must
    be one the ABI names, as a formal's must, and every **name** in its extents
    must be a **kernel argument**: the launch computes its scratch size before
    it enters the region, so an extent it cannot name there cannot be sized.
    A module constant is refused as an extent for that reason even though the
    body may read one elsewhere. A scalar local needs no scratch and is
    declared in the region body as before.

    An extent is a declared bound written as C, not a name copied over, so it
    need not be a single symbol. ``dimension(max_length,4)`` and
    ``dimension(nlayers+1)`` are both accepted, and both a formal and a local
    are read the same way. What is required is an integer expression over
    kernel arguments and literals using ``+``, ``-`` and ``*``: division is
    refused, because Fortran and C++ can disagree about the rounding of an
    integer division and a wrongly sized allocation would not announce
    itself. A lower bound other than 1 is refused as well -- the generated
    View subtracts a fixed 1 from each Fortran index -- so
    ``dimension(0:nlayers-1)`` is out of reach while ``dimension(1:nlayers)``
    is not. Every one of these is refused by :py:meth:`validate` rather than
    discovered by :py:meth:`apply`.

    It captures all information needed by the Kokkos backend before lowering
    the LFRic loop. The LFRic loop is then lowered so that its bound setup and
    halo-dirty calls are retained, and only the resulting generic loop is
    replaced.
    """

    #: Accesses a plain ``parallel_for`` over cells can honour. ``INC``,
    #: ``READINC`` and ``REDUCTION`` all need colouring or atomics.
    _SAFE_ACCESSES = (AccessType.READ, AccessType.WRITE, AccessType.READWRITE)

    def __str__(self):
        return "Capture a supported LFRic loop as a Kokkos launch"

    def validate(self, node, options=None, **kwargs):
        """Check that ``node`` matches the capture contract.

        The check is side-effect free: it predicts what :py:meth:`apply`
        would do, including whether each array section could be lowered,
        without altering the kernel schedule.

        :param node: the loop that is to be captured as a Kokkos region.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: a dictionary with options for transformations.
        :type options: Optional[Dict[str, Any]]
        :param kwargs: additional keyword arguments for the base
            :py:meth:`~psyclone.psyGen.Transformation.validate`.
        :type kwargs: unwrapped dict

        :raises TransformationError: if ``node`` is not an LFRicLoop, or if
            its bounds, its kernel's metadata, its body, its array sections,
            its shape enquiries, its formal arguments, its local arrays or the
            module constants it reads fall outside the contract stated in this
            class's description.
        """
        if not isinstance(node, LFRicLoop):
            raise TransformationError(
                "LFRicKokkosTrans expects an LFRicLoop but found "
                f"'{type(node).__name__}'.")

        self._validate_loop(node)
        kernel = node.kernels()[0]
        self._validate_kernel_metadata(kernel)
        schedule = self._schedule(kernel)
        self._validate_body(schedule)
        self._validate_sections(schedule)
        self._validate_bounds(schedule)
        self._validate_formals(schedule)
        self._validate_locals(schedule)
        self._constants(schedule)

    @staticmethod
    def _validate_loop(node):
        """Check the loop's own iteration contract.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`

        :raises TransformationError: if the loop is coloured, is not over
            cell columns, carries a halo depth, is not bounded by the owned
            cells, or does not hold exactly one kernel.
        """
        if node.loop_type or node.iteration_space != "cell_column":
            raise TransformationError(
                "LFRicKokkosTrans supports only an uncoloured cell-column "
                "loop.")
        if node.upper_bound_halo_depth is not None:
            raise TransformationError(
                "LFRicKokkosTrans does not support a halo depth.")
        # LFRicLoop does not currently expose its lower-bound name.
        # pylint: disable=protected-access
        if (node._lower_bound_name != "start" or
                node.upper_bound_name != "ncells"):
            raise TransformationError(
                "LFRicKokkosTrans supports only owned-cell bounds.")
        if len(node.kernels()) != 1:
            raise TransformationError(
                "LFRicKokkosTrans requires exactly one kernel in the loop.")

    @classmethod
    def _validate_kernel_metadata(cls, kernel):
        """Check the LFRic metadata of the kernel to be captured.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :raises TransformationError: if the kernel needs quadrature or
            evaluator data, is a CMA or inter-grid kernel, takes an argument
            that is neither a field nor a scalar, takes an access a
            cell-parallel launch cannot honour, takes a non-real field, uses a
            stencil, or writes to a field on a continuous space.
        """
        if kernel.qr_required or kernel.eval_shapes:
            raise TransformationError(
                "LFRicKokkosTrans does not support quadrature or evaluator "
                "data.")
        if kernel.cma_operation is not None:
            raise TransformationError(
                "LFRicKokkosTrans does not support CMA operators.")
        if kernel.is_intergrid:
            raise TransformationError(
                "LFRicKokkosTrans does not support inter-grid kernels.")

        discontinuous = LFRicConstants().VALID_DISCONTINUOUS_NAMES
        for argument in kernel.arguments.args:
            if argument.argument_type not in ("gh_field", "gh_scalar"):
                raise TransformationError(
                    "LFRicKokkosTrans supports only gh_field and gh_scalar "
                    f"arguments, found '{argument.argument_type}'.")
            if argument.access not in cls._SAFE_ACCESSES:
                raise TransformationError(
                    f"LFRicKokkosTrans cannot capture the '{argument.access}' "
                    f"access of '{argument.name}': a cell-parallel launch "
                    "would need colouring or atomics.")
            if argument.argument_type != "gh_field":
                continue
            if argument.intrinsic_type != "real":
                raise TransformationError(
                    f"LFRicKokkosTrans supports only real fields, but "
                    f"'{argument.name}' is {argument.intrinsic_type}.")
            if argument.stencil:
                raise TransformationError(
                    "LFRicKokkosTrans does not support stencil accesses.")
            if argument.access == AccessType.READ:
                continue
            space = argument.function_space.orig_name.lower()
            if space not in discontinuous:
                raise TransformationError(
                    f"LFRicKokkosTrans requires a discontinuous space for "
                    f"the written field '{argument.name}', but found "
                    f"'{space}': one cell's contribution could overwrite "
                    "another's.")

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
            # unbuilt -- evaluator shapes, stencils, CMA and inter-grid
            # kernels. Not being able to ask the question is a third outcome,
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

    @staticmethod
    def _lower_sections(schedule):
        """Replace every whole-column section with an explicit loop.

        Applied to the schedule :py:meth:`_schedule` returns, which
        :py:meth:`~psyclone.domain.lfric.LFRicKern.get_callees` caches so that
        transformations applied to a kernel persist. That is the intended
        idiom, so the lowering is done once here rather than repeated for
        every consumer of the schedule.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if an assignment holding a section
            cannot be lowered. :py:meth:`_validate_sections` predicts this by
            running this method over a copy, so reaching it from
            :py:meth:`apply` would mean that prediction had been skipped.
        """
        lowering = ArrayAssignment2LoopsTrans()
        for assignment in schedule.walk(Assignment):
            if assignment.walk(Range):
                lowering.apply(assignment)

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
    def _validate_locals(cls, schedule):
        """Check that every kernel-local array can be placed in team scratch.

        An automatic array is a per-cell temporary whose extent is a runtime
        value, so the region gives each team rank its own scratch View of it.
        Sizing that View is what the two refusals below protect: the element
        type has to be one the ABI names, and the extent has to be a value
        the launch already holds, which is a kernel argument.

        A module constant is refused as an extent even though
        :py:meth:`_constants` could import one, because the launch computes
        its scratch size before it enters the region.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if a local array's kind is not one
            :py:attr:`_C_TYPES` maps, or if one of its extents is not a
            kernel argument, so the scratch View could not be sized.
        """
        table = schedule.symbol_table
        names = {symbol.name for symbol in table.argument_list}
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

    def apply(self, node, options=None, **kwargs):
        """Generate C++ and replace ``node`` with the typed launch call.

        Unlike :py:meth:`validate`, this alters the kernel schedule: any
        array section it holds is lowered to an explicit loop, and every shape
        enquiry is replaced by the bound its declaration gives, before the
        region is described.

        :param node: the loop to capture as a Kokkos region.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: a dictionary with options for transformations.
        :type options: Optional[Dict[str, Any]]
        :param kwargs: additional keyword arguments for the base
            :py:meth:`~psyclone.psyGen.Transformation.apply`.
        :type kwargs: unwrapped dict

        :returns: the generated Kokkos C++ translation unit.
        :rtype: str

        :raises TransformationError: if ``node`` fails :py:meth:`validate`,
            if the PSy layer supplies a different number of actual arguments
            than the kernel has formals, or if the Kokkos backend cannot
            generate the region that validation predicted it could.
        """
        self.validate(node, options=options, **kwargs)
        kernel = node.kernels()[0]
        schedule = self._schedule(kernel)
        self._lower_sections(schedule)
        self._substitute_bounds(schedule)

        # KernCallArgList creates references to PSy-layer symbols. Ensure the
        # LFRic invoke has first specialised those symbols as DataSymbols.
        node.ancestor(InvokeSchedule).invoke.setup_psy_layer_symbols()
        argument_builder = KernCallArgList(kernel)
        argument_builder.generate()
        formals = schedule.symbol_table.argument_list
        actuals = [argument.copy()
                   for argument in argument_builder.psyir_arglist]
        if len(actuals) != len(formals):
            raise TransformationError(
                f"LFRicKokkosTrans expected {len(formals)} actual arguments "
                f"for '{kernel.name}' but the PSy layer supplies "
                f"{len(actuals)}.")

        # An actual that indexes into PSy-layer storage -- a dofmap sliced as
        # map(:,cell) -- is passed whole instead, and the region takes the
        # cell index itself. Everything else is already a plain reference.
        per_cell = set()
        for index, (formal, actual) in enumerate(zip(formals, actuals)):
            if not isinstance(actual, ArrayReference):
                continue
            per_cell.add(formal.name)
            actuals[index] = Reference(actual.symbol)

        constants = self._constants(schedule)
        region = KokkosRegion(
            name=self._region_name(schedule),
            schedule=schedule,
            cell_count=self._CELL_COUNT,
            arguments=self._region_arguments(schedule, per_cell, constants),
            kind_types=self._kind_types(schedule),
            scratch=self._local_arrays(schedule))
        try:
            cpp = KokkosWriter()(region)
        except (VisitorError, ValueError, TypeError) as err:
            raise TransformationError(
                f"LFRicKokkosTrans cannot express '{kernel.name}' in the "
                f"Kokkos backend: {err}") from err

        lowered_loop = node.lower_to_language_level()
        cell_count = lowered_loop.stop_expr.copy()
        routine = lowered_loop.ancestor(Routine)
        symbol_table = routine.symbol_table
        launch = self._launch_symbol(symbol_table, region)
        actuals.append(cell_count)
        actuals.extend(
            Reference(self._import_constant(symbol_table, name, container))
            for name, container, _ in constants)
        counter = lowered_loop.variable
        lowered_loop.replace_with(Call.create(launch, actuals))
        self._drop_unused_counter(routine, counter)
        return cpp


__all__ = ["LFRicKokkosTrans"]
