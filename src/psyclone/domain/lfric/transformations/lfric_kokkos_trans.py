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
from psyclone.lfric import LFRicHaloExchange
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

    A kind the ABI does not name is refused rather than guessed at -- a
    16-byte ``r_quad``, an undeclared precision, a module constant whose width
    the precision map does not carry.

    **A logical scalar is not one of them, because it crosses by conversion
    rather than by width.** The dummy is ``logical(c_bool), value`` and the
    call site wraps the actual in ``LOGICAL(..., c_bool)``, which is a
    conversion the compiler performs; the two kinds therefore need not agree,
    and no width assertion is generated for a logical because there is no
    width to assert. This matters beyond tidiness: LFRic's ``l_def`` is
    ``kind(.false.)`` and measures 4 bytes where PSyclone's precision map
    records 1, which is issue #1941. Nothing here reads that entry, so the
    prototype is correct at either value and a corrected #1941 would not
    change a line of what it generates.

    A logical **array** stays refused. Conversion is per value, and an array
    crosses by reference: a ``View<bool*>`` laid over ``logical(l_def)``
    storage would reinterpret 4-byte elements as 1-byte ones rather than
    convert them, which is the very failure conversion removes for a scalar.

    **A stencil is accepted by shape**, and the accepted shapes are
    :py:attr:`_SUPPORTED_STENCILS` -- ``cross2d`` alone. A 2-D stencil needs
    no argument machinery of its own: LFRic hands the kernel a sliced dofmap
    and a sliced size array, both of them array formals, so both become Views
    with the cell index appended exactly as the dofmap ``map_w3(:,cell)``
    already does. A 1-D or region stencil hands the size over as a *scalar*
    formal fed from ``field_stencil_size(cell)``, which would need a per-cell
    scalar argument kind that does not exist here, so those shapes are refused
    by name. A stencil also makes the PSy layer emit a halo exchange in front
    of the loop, which is lowered before the loop is replaced rather than
    after; see :py:meth:`_lower_halo_exchanges`. The matcher refusal above is
    unaffected: PSyclone's issue #928 leaves every stencil shape unbuilt, so a
    stencil kernel written as a generic interface is still refused there
    whatever its shape.

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

    **A local is also read for its name, not only its type.** The generated
    launch declares identifiers of its own in the scope the kernel body is
    generated into, and a kernel-local of the same name would shadow one and
    then overwrite it -- a wrong answer rather than a compile error. The cell
    count is one, and the team launch adds ``body``, ``league_size``,
    ``probe``, ``rank``, ``scratch_bytes``, ``team`` and ``team_size``, which
    are checked only for a kernel that has an automatic array to place. The
    launch *index* is the exception: it is renamed rather than refused,
    because a kernel declaring ``cell`` is a real GungHo shape and the fix is
    one name in two places rather than seven threaded through two launch
    shapes.

    **A constant the body reads reaches the region one of three ways.** A
    module-level ``parameter`` declared beside the kernel with a literal value
    -- ``integer(kind=i_def), parameter :: nfaces = 4`` -- is written into the
    region as that value. It has to be: a kernel module is ``private`` by
    default and publishes only its ``_code`` routine, so importing the name
    into the PSy layer would not compile. One declared with anything other
    than a literal, an array ``parameter`` among them, is refused. A constant
    *imported* from another module is passed by value instead, which needs its
    kind, and so needs the source of its container on PSyclone's module search
    path; without that it is refused with a message naming the module to add
    rather than a guess at its width. Last, a name appearing only as an
    intrinsic's ``kind`` argument -- the ``r_def`` of ``real(x, r_def)`` -- is
    neither: it names a type, the cast consumes it, and the region carries the
    width rather than the name.

    It captures all information needed by the Kokkos backend before lowering
    the LFRic loop. The LFRic loop is then lowered so that its bound setup and
    halo-dirty calls are retained, and only the resulting generic loop is
    replaced.
    """

    #: Accesses a plain ``parallel_for`` over cells can honour. ``INC``,
    #: ``READINC`` and ``REDUCTION`` all need colouring or atomics.
    _SAFE_ACCESSES = (AccessType.READ, AccessType.WRITE, AccessType.READWRITE)

    #: Identifiers the generated launch declares in the scope the kernel
    #: body is generated into. A kernel-local of the same name would
    #: shadow the launch's own and then overwrite it, which is a wrong
    #: answer rather than a compile error. The cell index is absent
    #: because it is renamed instead; see
    #: :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_index`.
    #: These seven belong to the team launch only, so they are checked only
    #: for a kernel that has an automatic array to place;
    #: :py:attr:`_CELL_COUNT` is declared by both shapes and is always
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
            stencil shape outside :py:attr:`_SUPPORTED_STENCILS`, or writes to
            a field on a continuous space.
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
                shape = str(argument.stencil.name).lower()
                if shape not in cls._SUPPORTED_STENCILS:
                    raise TransformationError(
                        f"LFRicKokkosTrans supports the "
                        f"{', '.join(cls._SUPPORTED_STENCILS)} stencil shape "
                        f"only, but '{argument.name}' has '{shape}'.")
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

        :raises TransformationError: if the kernel declares a local named
            :py:attr:`_CELL_COUNT`, which both launch shapes declare.
        :raises TransformationError: if the kernel has an automatic array and
            declares a local named in :py:attr:`_GENERATED_NAMES`, which the
            team launch that array selects declares.
        :raises TransformationError: if a local array's kind is not one
            :py:attr:`_C_TYPES` maps, or if one of its extents is not a
            kernel argument, so the scratch View could not be sized.
        """
        table = schedule.symbol_table
        names = {symbol.name for symbol in table.argument_list}

        locals_ = list(table.automatic_datasymbols)
        generated = {cls._CELL_COUNT}
        if any(symbol.is_array for symbol in locals_):
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

    @classmethod
    def _as_c_bool(cls, actual, symbol_table):
        """Wrap one actual argument in a conversion to ``logical(c_bool)``.

        This is what puts a Fortran ``logical`` on the C ABI without either
        side knowing the other's width. The dummy is ``logical(c_bool),
        value``; the actual is whatever kind LFRic declared, typically
        ``l_def``; and ``LOGICAL(x, c_bool)`` is a standard conversion the
        compiler performs, not a reinterpretation of storage. Neither side
        consults the precision map, which is why PSyclone issue #1941 --
        recording ``l_def`` as 1 byte where it is 4 -- cannot affect the
        result.

        :param actual: the argument expression to convert, already detached
            from the tree or freshly built.
        :type actual: :py:class:`psyclone.psyir.nodes.DataNode`
        :param symbol_table: the PSy-layer routine's table, which gains the
            ``c_bool`` import if it does not already carry one.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`

        :returns: the conversion, ready to stand in the actual's place.
        :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        """
        c_bool = cls._import_constant(symbol_table, "c_bool", "iso_c_binding")
        return IntrinsicCall.create(
            IntrinsicCall.Intrinsic.LOGICAL,
            [actual, ("kind", Reference(c_bool))])

    @staticmethod
    def _lower_halo_exchanges(node):
        """Lower every halo exchange in ``node``'s invoke, before ``node`` is.

        A halo exchange does not know its own depth: it computes one by
        walking forward for the accesses that read the field it exchanges, and
        those accesses are LFRic kernel arguments carrying LFRic metadata.
        :py:meth:`apply` is about to replace the loop holding them with a
        plain :py:class:`~psyclone.psyir.nodes.Call`, which carries none, so
        an exchange lowered afterwards finds no reader at all and PSyclone
        raises :py:class:`~psyclone.errors.InternalError` rather than
        generating a wrong depth.

        Lowering the exchanges first is the order whole-container lowering
        would have used anyway -- an exchange precedes the loop it feeds, and
        lowering runs in schedule order -- so this restores that order rather
        than choosing a new one. It does nothing for a schedule that has no
        exchange, which is every region captured before stencils.

        :param node: the loop about to be captured, used only to reach the
            invoke schedule containing it. A node with no
            :py:class:`~psyclone.psyGen.InvokeSchedule` ancestor, as a unit
            test's bare schedule has, is left alone.
        :type node: :py:class:`~psyclone.domain.lfric.LFRicLoop`

        """
        schedule = node.ancestor(InvokeSchedule)
        if schedule is None:
            return
        for exchange in schedule.walk(LFRicHaloExchange):
            exchange.lower_to_language_level()

    def apply(self, node, options=None, **kwargs):
        """Generate C++ and replace ``node`` with the typed launch call.

        Unlike :py:meth:`validate`, this alters the kernel schedule: any
        array section it holds is lowered to an explicit loop, every shape
        enquiry is replaced by the bound its declaration gives, and every
        module-level ``parameter`` it reads is replaced by its value, before
        the region is described.

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
        self._substitute_constants(schedule)

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
        # The launch index shares a C++ scope with the kernel's own
        # declarations, so a kernel declaring 'cell' would collide with it.
        # Spelt from the dataclass default so the two cannot drift: a kernel
        # that has not taken the name still generates 'cell'.
        cell_index = schedule.symbol_table.next_available_name(
            KokkosRegion.cell_index)
        region = KokkosRegion(
            name=self._region_name(schedule),
            schedule=schedule,
            cell_count=self._CELL_COUNT,
            cell_index=cell_index,
            arguments=self._region_arguments(
                schedule, per_cell, constants, cell_index),
            kind_types=self._kind_types(schedule),
            scratch=self._local_arrays(schedule))
        try:
            cpp = KokkosWriter()(region)
        except (VisitorError, ValueError, TypeError) as err:
            raise TransformationError(
                f"LFRicKokkosTrans cannot express '{kernel.name}' in the "
                f"Kokkos backend: {err}") from err

        self._lower_halo_exchanges(node)
        lowered_loop = node.lower_to_language_level()
        cell_count = lowered_loop.stop_expr.copy()
        routine = lowered_loop.ancestor(Routine)
        symbol_table = routine.symbol_table
        launch = self._launch_symbol(symbol_table, region)
        actuals.append(cell_count)
        actuals.extend(
            Reference(self._import_constant(symbol_table, name, container))
            for name, container, _ in constants)
        # region.arguments is the formals, then the cell count, then the
        # constants -- which is exactly the order 'actuals' is in once both
        # appends above have run. The two are therefore index-aligned, and one
        # loop covers a logical formal and an imported logical constant alike.
        # That alignment is what makes this correct and it is not visible from
        # the loop itself.
        for index, argument in enumerate(region.arguments):
            if argument.c_type == self._C_LOGICAL_TYPE:
                actuals[index] = self._as_c_bool(actuals[index], symbol_table)
        counter = lowered_loop.variable
        lowered_loop.replace_with(Call.create(launch, actuals))
        self._drop_unused_counter(routine, counter)
        return cpp


__all__ = ["LFRicKokkosTrans"]
