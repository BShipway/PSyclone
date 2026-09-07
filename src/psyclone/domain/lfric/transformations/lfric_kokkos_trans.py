# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Capture a supported LFRic loop as a Kokkos launch."""

from psyclone.domain.lfric import KernCallArgList, LFRicLoop
from psyclone.domain.lfric.transformations.lfric_kokkos_bounds_mixin import (
    LFRicKokkosBoundsMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_call_mixin import (
    LFRicKokkosCallMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_constants_mixin \
    import LFRicKokkosConstantsMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_contract_mixin import (
    LFRicKokkosContractMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_types_mixin import (
    LFRicKokkosTypesMixin)
from psyclone.errors import GenerationError
from psyclone.lfric import LFRicHaloExchange
from psyclone.psyGen import InvokeSchedule, Transformation
from psyclone.psyir.backend.kokkos import KokkosRegion, KokkosWriter
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import (
    ArrayReference, Assignment, Call, IntrinsicCall, Literal, Loop, Range,
    Reference, Routine)
from psyclone.psyir.tools import DependencyTools
from psyclone.psyir.transformations import (
    ArrayAssignment2LoopsTrans, TransformationError)


class LFRicKokkosTrans(LFRicKokkosContractMixin, LFRicKokkosTypesMixin,
                       LFRicKokkosBoundsMixin, LFRicKokkosCallMixin,
                       LFRicKokkosConstantsMixin, Transformation):
    """Replace one supported LFRic cell-column loop with a C ABI call.

    The transformation recognises a kernel shape rather than a named kernel:
    an uncoloured owned-cell loop over a single kernel whose arguments are
    fields, scalars and LMA operators, whose written fields are on
    discontinuous spaces, and whose formals and referenced module constants
    all map onto the ``int``/``float``/``double`` ABI the Kokkos backend
    emits. Every part of the generated region -- its name, its C signature,
    its Views and the ``bind(C)`` interface the PSy layer calls through -- is
    derived from that kernel, so a second kernel needs no change here.

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

    **A formal declared with no kind at all is accepted if it is an integer
    or a logical, and refused if it is a real.** GungHo writes plenty of
    both: ``logical, intent(in) :: include_surface``, and the whole of the
    integer housekeeping ``nlayers, ndf_w0, undf_w0, map_w0`` that LFRic's
    argument ordering supplies. The two are admitted for different reasons.
    A logical needs no width at all, by the conversion above, so an unnamed
    logical kind is the ``l_def`` case with the name taken off -- ``l_def``
    *is* ``kind(.false.)``, so the two declarations say the same thing.

    An integer does cross at a width, and the default integer's width is
    named by no kind parameter the precision map could be asked about. It is
    therefore measured rather than assumed: the generated interface carries
    ``storage_size(1) == storage_size(1_c_int)`` as a compile-time assertion
    like any other kind's, so a build whose default integer is not ``c_int``
    fails to compile rather than losing half of every value. The assertion is
    labelled ``assert_kind_default_integer``, and that name is this
    transformation's own -- ``constants_mod`` declares no such kind, being
    unnamed being the whole of what makes it the default -- so it is written
    into no ``use`` statement.

    A **real** declared with no kind stays refused, and the asymmetry is
    deliberate. LFRic names a kind on every real it means: ``r_def``,
    ``r_solver``, ``r_single`` and ``r_tran`` are all in use and all
    different, so the kind is the whole of what a real declaration says about
    its width, and one that says nothing is more likely an oversight than a
    default. A **width stated in place of a kind name** -- ``integer*8``, or
    ``real(kind=8)`` -- is refused too, for all three intrinsics: that
    declaration did say which width it wanted, and reading it as the default
    would be the silent narrowing this whole contract exists to prevent.

    **An LMA operator is accepted; a CMA operator is not.** An LMA
    operator reaches the kernel as ``ncell_3d`` and a rank-3 array
    ``dimension(ncell_3d, ndf1, ndf2)`` holding every cell's local stencil
    end to end. Every extent is a formal of its own, so the operator becomes
    an ordinary read-only View and needs no argument machinery: what it needs
    is the cell. A CMA operator stays refused, as before, because a banded
    matrix carries its own bandwidth and indexing arguments that this region
    has no way to describe.

    **The kernel's cell argument is declared rather than passed.** LFRic gives
    a leading ``cell`` formal to exactly the kernels that take an operator,
    because a kernel finds its own slice of a local stencil arithmetically --
    ``ik = (cell - 1) * nlayers + 1`` -- rather than being handed the slice.
    The launch already knows which cell it is on, so the region declares
    ``const int cell = cell_1 + 1;`` from its own index and the formal never
    reaches the generated signature. Passing it instead is what makes this
    worth stating: it would become a launch parameter fixed for the whole
    region, and the actual the PSy layer supplies is its own loop counter,
    which lowering removes. Every cell would then compute ``ik`` from an
    unassigned variable -- code that compiles, links, runs and is wrong.

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
    only at runtime -- is placed in Kokkos team scratch, so that the cells
    sharing a team do not share a temporary. The region is then launched over
    a ``TeamPolicy`` rather than a ``RangePolicy``.

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
    itself. Every one of these is refused by :py:meth:`validate` rather than
    discovered by :py:meth:`apply`.

    **A declared lower bound need not be 1.** ``dimension(0:nlayers-1)`` and
    ``dimension(-nlayers:nlayers)`` are accepted alongside
    ``dimension(nlayers)``: the View is sized by the span the declaration
    gives, ``ub - lb + 1``, and every subscript of that array has ``lb``
    subtracted from it on the way to the zero-based element, so
    ``u_e(k)`` over a local declared ``dimension(0:nlayers)`` becomes
    ``u_e((k - 0))``. The lower bound must satisfy the same grammar as the
    upper -- an integer expression over kernel arguments and literals using
    ``+``, ``-`` and ``*`` -- and is refused on the same terms when it does
    not. A bound of 1 renders exactly the source it rendered before this was
    accepted, the span folding back to the upper bound alone.

    The subtraction is written out even where it is zero, because it is
    applied in one place -- the back-end's generation of an array accessor --
    and a subscript that escaped it would compile and give a wrong answer
    rather than fail. What is gained by omitting it is what the C++ optimiser
    removes anyway.

    **A loop inside the kernel body may be spread over the team**, and which
    loops those are is PSyclone's own judgement rather than this
    transformation's: each is put to
    :py:meth:`~psyclone.psyir.tools.DependencyTools.can_loop_be_parallelised`,
    and one it accepts becomes a ``Kokkos::TeamVectorRange`` over the team's
    members. Only the outermost of an accepted nest is taken, because a
    ``TeamVectorRange`` may not be nested inside itself, and a loop whose step
    is not 1 is left alone, because that range gives every member a unit
    stride. In practice these are the level loops: a GungHo kernel's outer
    loop runs over a column's levels, and the loops that survive the analysis
    are the ones whose iterations touch disjoint elements rather than sweeping
    a recurrence.

    A kernel with such a loop is launched over one team per cell -- the cell
    is the team's league rank -- **whether or not it has an automatic array**.
    The two selections are separate, and the region shows it: a kernel with
    scratch and no acceptable loop keeps the flat launch, where the team is a
    way of owning a per-member temporary and its scratch is per member; a
    kernel with both takes the one-team-per-cell shape and its scratch becomes
    per team, shared by the members working on that column. A kernel with
    neither keeps the ``RangePolicy``.

    Everything outside a spread loop runs on **every** member of the team, on
    that member's own copy of the scalar locals. That is harmless for a scalar
    but not for an array, so an array write outside a spread loop is made by
    one member under ``Kokkos::single`` and published to the rest by a team
    barrier before the next statement reads it. A barrier also follows each
    spread loop, unconditionally: deciding whether a later reader needs it is
    a second analysis this transformation does not do.

    ``"team_size"`` is the one option, an optional positive ``int``. Absent,
    the policy asks for ``Kokkos::AUTO`` and the backend sizes the team --
    which on the OpenMP backend is **one member**, so on a host build the
    leagues carry all of the parallelism and a spread loop runs serially. A
    host build reaches the team-level concurrency only by naming a size here.

    **A local is also read for its name, not only its type.** The generated
    launch declares identifiers of its own in the scope the kernel body is
    generated into, and a kernel-local of the same name would shadow one and
    then overwrite it -- a wrong answer rather than a compile error. The cell
    count is one, and the two team launches add ``body``, ``league_size``,
    ``probe``, ``rank``, ``scratch_bytes``, ``team`` and ``team_size``, which
    are checked for a kernel that has an automatic array to place **or** a
    loop to spread -- either reaches a launch that declares them. The launch
    *index* is the exception: it is renamed rather than refused, because a
    kernel declaring ``cell`` is a real GungHo shape and the fix is one name
    in two places rather than seven threaded through two launch shapes.

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

    **A loop this transformation leaves behind may still be transformed
    afterwards**, colouring included, even though capturing forces the
    Invoke's PSy-layer symbols to be set up early. The region's actual
    arguments are built by
    :py:class:`~psyclone.domain.lfric.KernCallArgList`, which reads symbols
    that
    :py:meth:`~psyclone.domain.lfric.LFRicInvoke.setup_psy_layer_symbols`
    specialises, so that pass has to run here rather than at code generation:
    by then the kernel that would supply them has been removed from the tree.
    It is not idempotent, so the call code generation would otherwise make is
    suppressed, and whatever a later transformation has since made necessary
    is added instead by
    :py:meth:`~psyclone.domain.lfric.LFRicInvoke.complete_psy_layer_symbols`.
    Colouring is the case that needs it:
    :py:class:`~psyclone.domain.lfric.transformations.LFRicColourTrans`
    creates the colourmap symbols and replaces one loop with two, and without
    that completion the colourmaps would be declared and never assigned and
    the new loops would keep the placeholder bounds
    :py:class:`~psyclone.domain.lfric.LFRicLoop` gave them.

    The one thing completion cannot repair is a colourmap look-up in an
    Invoke that has no mesh object -- one built without distributed memory
    whose loops were all uncoloured when the capture ran -- because the mesh
    is obtained from a kernel argument that capture has removed. That raises
    :py:class:`~psyclone.errors.GenerationError` at code generation, naming
    the Invoke, rather than emitting a look-up on an unassigned pointer.
    Colouring such an Invoke before capturing it is refused by
    :py:meth:`validate` and colouring it afterwards by this, so the case is
    reported either way round.
    """

    #: The option naming the team size the hierarchical launch asks for.
    #: Absent, the launch writes ``Kokkos::AUTO`` and lets the backend size
    #: the team; on the OpenMP backend that is one member, so a host build
    #: reaches the team-level concurrency only by setting this.
    _TEAM_SIZE_OPTION = "team_size"

    def __str__(self):
        return "Capture a supported LFRic loop as a Kokkos launch"

    def validate(self, node, options=None, **kwargs):
        """Check that ``node`` matches the capture contract.

        The check is side-effect free: it predicts what :py:meth:`apply`
        would do, including whether each array section could be lowered,
        without altering the kernel schedule.

        :param node: the loop that is to be captured as a Kokkos region.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: a dictionary with options for transformations. The
            one read here is ``"team_size"``; see :py:meth:`apply`.
        :type options: Optional[Dict[str, Any]]
        :param kwargs: additional keyword arguments for the base
            :py:meth:`~psyclone.psyGen.Transformation.validate`.
        :type kwargs: unwrapped dict

        :raises TransformationError: if ``node`` is not an LFRicLoop, or if
            its bounds, its kernel's metadata, its body, its array sections,
            its shape enquiries, its formal arguments, its local arrays or the
            module constants it reads fall outside the contract stated in this
            class's description.
        :raises TransformationError: if the ``"team_size"`` option is neither
            absent nor a positive integer.
        """
        if not isinstance(node, LFRicLoop):
            raise TransformationError(
                "LFRicKokkosTrans expects an LFRicLoop but found "
                f"'{type(node).__name__}'.")

        team_size = (options or {}).get(self._TEAM_SIZE_OPTION)
        if team_size is not None and (
                isinstance(team_size, bool) or not isinstance(team_size, int)
                or team_size <= 0):
            raise TransformationError(
                f"LFRicKokkosTrans' '{self._TEAM_SIZE_OPTION}' option must be "
                f"a positive integer, but found '{team_size}'.")

        self._validate_loop(node)
        kernel = node.kernels()[0]
        self._validate_kernel_metadata(kernel)
        schedule = self._schedule(kernel)
        self._validate_body(schedule)
        self._validate_sections(schedule)
        self._validate_bounds(schedule)
        self._validate_formals(schedule)
        # Which names the launch reserves depends on which launch is selected,
        # and that is decided by the loops apply() will spread over the team.
        # Those are asked of a copy carrying the rewrites apply() makes,
        # because the lowering is itself a producer of loops: a kernel whose
        # only parallelisable loop is the one a section lowers to has none at
        # all until the copy is lowered. The two predicates above have already
        # shown that both rewrites succeed on this schedule.
        probe = schedule.copy()
        self._lower_sections(probe)
        self._substitute_bounds(probe)
        self._validate_locals(schedule, self._parallel_loops(probe))
        self._constants(schedule)

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
            if tools.can_loop_be_parallelised(loop):
                chosen.append(loop)
        return tuple(chosen)

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
        the region is described. The loops to spread over the team are chosen
        after all three, because the first two create loops.

        :param node: the loop to capture as a Kokkos region.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: a dictionary with options for transformations.
            ``"team_size"`` sets the team the hierarchical launch asks for, as
            a positive integer; absent, the launch writes ``Kokkos::AUTO``.
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
        parallel_loops = self._parallel_loops(schedule)

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

        cell_position = self._cell_position(kernel, node, formals, actuals)
        if cell_position is not None:
            # Both lists, together. They are walked in step below -- and
            # 'formals' is what describes the generated signature -- so
            # dropping the actual alone would attribute every later actual to
            # the formal before it, and the dofmaps, being the ones detected
            # by the shape of their actual, would silently lose the cell
            # dimension that makes them per-cell.
            formals = formals[1:]
            del actuals[0]

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
            cell_position=cell_position,
            arguments=self._region_arguments(
                formals, per_cell, constants, cell_index),
            kind_types=self._kind_types(schedule),
            scratch=self._local_arrays(schedule),
            parallel_loops=parallel_loops,
            team_size=(options or {}).get(self._TEAM_SIZE_OPTION))
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
            Reference(self._import_constant(
                symbol_table, name, container, orig_name))
            for name, container, orig_name, _ in constants)
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
