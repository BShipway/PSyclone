# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Capture a supported LFRic loop as a Kokkos launch."""

from psyclone.domain.lfric import LFRicLoop
from psyclone.domain.lfric.transformations.lfric_kokkos_argument_mixin \
    import LFRicKokkosArgumentMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_bounds_mixin import (
    LFRicKokkosBoundsMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_call_mixin import (
    LFRicKokkosCallMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_constants_mixin \
    import LFRicKokkosConstantsMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_contract_mixin import (
    LFRicKokkosContractMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_interface_mixin \
    import LFRicKokkosInterfaceMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_intrinsic_mixin \
    import LFRicKokkosIntrinsicMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_inline_mixin import (
    LFRicKokkosInlineMixin)
from psyclone.domain.lfric.transformations.lfric_kokkos_schedule_mixin \
    import LFRicKokkosScheduleMixin
from psyclone.domain.lfric.transformations.lfric_kokkos_types_mixin import (
    LFRicKokkosTypesMixin)
from psyclone.psyGen import Transformation
from psyclone.psyir.backend.kokkos import KokkosWriter
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.transformations import TransformationError


# Ten mixins and Transformation, which is one contract split by subject
# rather than eleven layers of behaviour: every base but the last holds only
# private helpers, and none of them overrides anything.
# pylint: disable-next=too-many-ancestors
class LFRicKokkosTrans(LFRicKokkosContractMixin, LFRicKokkosTypesMixin,
                       LFRicKokkosArgumentMixin, LFRicKokkosBoundsMixin,
                       LFRicKokkosCallMixin,
                       LFRicKokkosConstantsMixin, LFRicKokkosInlineMixin,
                       LFRicKokkosInterfaceMixin,
                       LFRicKokkosIntrinsicMixin, LFRicKokkosScheduleMixin,
                       Transformation):
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
    cannot model -- a stencil or a CMA kernel, which PSyclone's issue #928
    leaves unbuilt -- is a third refusal, kept distinct from
    finding no match because a question that cannot be asked has not been
    answered "no".

    An evaluator shape was one of those and is no longer. Rather than reach
    past the matcher and compare the actual arguments here, citing #928 as the
    reason, the matcher itself was taught what an evaluator implies: one
    rank-3 basis per target space, extents of ``(dim, ndf, ndf of the
    target)``, with no point count and no weights because an evaluator carries
    no quadrature rule. That is a smaller answer than it sounds -- an
    evaluator asks for no callback the class does not already have -- and it
    keeps the question in the one place that asks it, so a kernel is selected
    between on its precisions whatever shape its basis has.

    It also removes a refusal that was a misreading rather than a mismatch.
    Metadata cannot fix a basis's first extent for an ``any_space`` or an
    ``any_discontinuous_space``, which is PSyclone's issue #461, and a matcher
    that could not produce that extent reported the members as matching no
    algorithm precision at all -- an answer about precisions it had never
    compared. The extent is instead named with a variable, exactly as the PSy
    layer names it, and it is not one of the things the matcher compares.

    An inter-grid shape was one of those too, and was taught the same way:
    the cell map with its two per-cell counts and the fine mesh's cell count,
    and a whole dofmap for a field on the fine mesh where a per-cell one
    serves a field on the coarse. Both are what the PSy layer already passes,
    so what the matcher gained is a description rather than a decision.

    **A field's data may be real or integer.** LFRic's ``integer_field_type``
    has ``field_type``'s proxy shape with ``integer(i_def)`` data, and the
    region's element type follows the argument's own declaration rather than
    being fixed when the field ABI is written, so a ``gh_integer`` field
    crosses as a ``View<int*>`` where a ``gh_real`` one crosses as a
    ``View<double*>``. Nothing else about the field moves with the intrinsic:
    the proxy member the PSy layer reads, the ``ndf`` and ``undf`` formals,
    the dofmap and the launch belong to the function space. A kernel taking
    both is one region carrying both widths, each formal at its own, and the
    arithmetic between two integer field values stays integer -- the writer
    types its expressions from their operands, so a quotient of two integer
    Views truncates as the Fortran it was generated from does.

    A kind the ABI does not name is refused rather than guessed at -- a
    16-byte ``r_quad``, an undeclared precision, a module constant whose width
    the precision map does not carry. An integer field is not exempt: its
    declaration is read like every other formal's, so one at a width the ABI
    has no C type for is refused naming the kind rather than written as
    ``int`` because its intrinsic was admitted.

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

    **An inter-grid kernel spans two meshes and still launches over one.**
    Such a kernel takes a field on a coarse mesh and one on a fine mesh, and
    LFRic runs it once per *coarse* cell: the cell map ``cell_map(:,:,cell)``
    names the ``ncell_f_per_c_x`` by ``ncell_f_per_c_y`` fine cells that cell
    refines into, and the body reaches them itself. So the launch is the flat
    one it would have been over a single mesh, and the second mesh reaches
    the region as data and never as an iteration space -- ``ncell_f`` is the
    extent the fine mesh's dofmap is strided by, not a bound anything is
    launched over.

    Nothing had to be built for that. The cell map is an actual the PSy layer
    slices by cell, so the per-cell rule turns it into a rank-3 View like any
    other sliced dofmap; the three counts beside it are scalars; the fine
    mesh's dofmap arrives whole, ``map_f(:,:)``, which the same rule leaves
    whole because it is not sliced; and the coarse mesh is what LFRic's own
    loop bound already names. What stood in the way was a refusal, and
    removing it is what accepts these kernels.

    **A region may iterate into the halo, and the exchange stays in the PSy
    layer.** The launch's upper bound is a formal the region reads and never
    writes, filled by the PSy layer with the loop's own stop expression, so
    what the bound *means* is settled outside the region and only its value
    crosses. LFRic writes three such expressions and all three are accepted:
    ``mesh%get_last_halo_cell(1)`` for a loop taken to the first halo depth,
    ``field_proxy%vspace%get_last_dof_annexed()`` for a dof loop taken over
    the annexed dofs, and ``mesh%get_last_halo_cell(depth)`` for a depth
    computed at runtime -- that depth is evaluated where it already was, and
    the region never sees it.

    Every per-cell View is sliced to the same formal, so a region running into
    the halo describes the cells it runs over rather than the owned ones. What
    the accepted bounds have in common is that each counts consecutively from
    the first cell or dof, which is :py:attr:`_COUNTED_BOUNDS`; a loop whose
    *lower* bound is shifted -- ``cell_halo_start``, which runs the halo alone
    -- is still refused, the launch having no lower-bound formal to fill.

    The halo exchange itself does not move. The PSy layer emits it in front of
    the loop and it is lowered there, in front of the call to the region; see
    :py:meth:`_lower_halo_exchanges`. Nothing is exchanged inside a region.

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
    :py:attr:`_SUPPORTED_STENCILS` -- ``cross``, ``cross2d`` and ``region``.
    A 2-D stencil needs no argument machinery of its own: LFRic hands the
    kernel a sliced dofmap and a sliced size array, both of them array
    formals, so both become Views with the cell index appended exactly as the
    dofmap ``map_w3(:,cell)`` already does.

    A 1-D or region stencil hands the size over as a *scalar* formal fed from
    ``field_stencil_size(cell)``. That is one cell's value and a region runs
    every cell at once, so the size crosses the ABI as the whole rank-1 array
    and the body subscripts it by the launch's own cell -- a scalar formal
    made per-cell, which is what the 2-D shape's size array already is. The
    dofmap beside it is declared ``dimension(ndf, stencil_size)``, over that
    per-cell value, while LFRic allocates it uniformly. A View's extents fix
    its strides, so a per-cell value cannot size one: the region carries the
    dofmap's storage extent beside it as a scalar of its own, measured by the
    PSy layer as ``SIZE(field_stencil_dofmap, dim=2)``, and sizes the View and
    any automatic array declared over the stencil from that instead.

    ``xory1d`` is refused by name. It carries a direction argument on top of
    the 1-D size, chosen per cell in the algorithm layer, and nothing here
    describes it.

    A stencil also makes the PSy layer emit a halo exchange in front of the
    loop, which is lowered before the loop is replaced rather than after; see
    :py:meth:`_lower_halo_exchanges`. The matcher refusal above is unaffected:
    PSyclone's issue #928 leaves every stencil shape unbuilt, so a stencil
    kernel written as a generic interface is still refused there whatever its
    shape.

    **A basis is accepted by shape** too, and the accepted shapes are
    :py:attr:`_SUPPORTED_SHAPES` -- ``gh_quadrature_XYoZ`` and
    ``gh_evaluator``. Neither needs argument machinery of its own. XYoZ
    quadrature adds two point counts, two weight arrays and one basis array
    per function space that asked for one, shaped
    ``(dim, ndf, np_xy, np_z)``; an evaluator adds no rule at all, tabulating
    the basis at the nodal points of a target function space to give
    ``(dim, ndf, ndf of the target)`` and no weights. Every one of those is an
    argument the PSy layer has computed before the loop and every extent of it
    is a formal of the same kernel, so the existing scalar and View
    descriptions cover them whole. A kernel may name both shapes, in which
    case each space it declares carries a basis array per shape; each shape is
    checked on its own so that a refusal names the one that is not modelled
    rather than the whole set.

    Face and edge quadrature are refused by name. They carry a face or edge
    count and a single point count in place of the XYoZ pair, and while the
    same descriptions look as though they would cover those too, nothing here
    has been measured against the model for them. Refusing by name says that;
    accepting on the strength of the resemblance would not.

    **A called subroutine is inlined, not called.** The generated region is
    a C++ function and there is no Fortran for it to call into, so a kernel
    that calls a helper has that helper's statements made its own before
    anything else looks at the body. The rewrite is
    :py:class:`~psyclone.psyir.transformations.InlineTrans`, applied one call
    at a time and repeated until the body makes none: a helper that calls a
    second helper leaves that second call behind in the statements it
    contributes, so one pass would not be enough. It runs before every other
    rewrite here, because a callee brings its own loops, its own locals and
    its own array sections in with it and each of those is then judged like
    the kernel's own.

    The repetition is bounded by ``LFRicKokkosInlineMixin._INLINE_LIMIT``,
    eight calls into one kernel body, and the bound is load-bearing rather
    than defensive: ``InlineTrans`` has no recursion check, so a routine that
    calls itself is substituted into itself for as long as it is asked.
    Reaching the bound is a refusal naming the routine still to be inlined.

    **A callee out of scope is refused rather than guessed at.** In scope are
    a procedure of the kernel's own module, and a procedure of a module the
    kernel names in a ``use`` whose source PSyclone can read -- the latter
    is first brought into the kernel's Container by
    :py:class:`~psyclone.domain.common.transformations.\
KernelModuleInlineTrans`.
    A callee whose module is not on the search path is not: PSyclone has a
    name for it and nothing else, and there is no body to inline.

    Being in scope is not being inlinable, and the rest of the judgement is
    PSyclone's rather than this transformation's: a callee reading data
    private to its own module, one whose declarations depend on an argument
    the call site writes to before calling, one whose actual and formal types
    do not agree, one holding a CodeBlock. Each is refused in ``InlineTrans``'
    own words with the call named, because those words say what to fix and a
    paraphrase would say less. So is a name that turns out not to be a call
    at all: an indexed reference the kernel's own file does not settle the
    meaning of is read as one by the frontend, and resolving it can reach a
    datum and raise :py:exc:`TypeError` rather than refuse. That too is a
    refusal here, so that :py:meth:`validate` declines a kernel it cannot
    capture instead of raising out of PSyclone.

    **An array-valued assignment is lowered to an explicit loop.** Two
    shapes reach the lowering. A whole-column array section such as
    ``a(i:j) = ...``, which the finite-volume kernels use to assign a column
    as a unit, is one; a whole-array assignment naming no section at all --
    ``vector = 0.0_r_def``, where ``vector`` is declared with a shape -- is
    the other, and it is expanded into a section by
    :py:class:`~psyclone.psyir.transformations.Reference2ArrayRangeTrans`
    first, which makes explicit the section it already meant.
    :py:class:`~psyclone.psyir.transformations.ArrayAssignment2LoopsTrans`
    then rewrites both into loops, before the region is described.

    The generated region has no way to say ``a(i:j)``, so a shape that
    transformation refuses is refused here too, with its reason quoted. The
    refusal that arises in GungHo is a loop-carried dependency, which no
    order of generated loops could honour.

    **An array-valued intrinsic is generated, not refused.** ``MATMUL``,
    ``DOT_PRODUCT``, ``SUM``, ``MINVAL``, ``MAXVAL`` and ``TRANSPOSE`` are
    written by the backend as loop nests over the destination, and no array
    temporary is created for any of them: one element of a contraction is a
    scalar reduction, one element of a transpose is an index swap, so there
    is nothing to hold between the two. ``MATMUL(TRANSPOSE(m), x)`` follows
    from that rather than needing a case of its own. What is refused is a
    fold this shape cannot express -- a ``dim`` that is not a literal or
    names no dimension the operand has, a ``mask``, a reduction nested
    directly inside another -- and an operand that is neither a whole array
    nor a section of one. ``RESHAPE`` is generated only where the source is
    rank 1 and the shape is a literal constructor, which makes it an index
    map over storage the two languages already agree about.

    A section that is not in an assignment at all is judged by where it is
    instead. One that is an actual argument of a call is left to the call:
    inlining the callee takes the argument away with it, so the section is
    gone before this rule could have refused it, and where the callee cannot
    be inlined that is the refusal worth reporting. One in the bounds of an
    ``ALLOCATE`` is read as the shape it states, below. Anywhere else it is
    beyond what lowering can reach and is refused before the backend sees it.

    **An array constructor fills an array; it is not a value.** A kernel
    writing ``v_dot_n = (/ -1.0, 1.0, 1.0, -1.0 /)``, or filling one
    full-extent dimension of an array as ``vert_vec(:,qp1,qp2) = (/ ... /)``,
    is generated as one assignment per element, into the array the kernel has
    already declared and from the origin that declaration gives. A braced
    initialiser is not the alternative it looks like: C accepts one only on a
    declaration, and the array is declared before the statement is reached.
    Anywhere else -- an actual argument, an operand of an expression, a
    constructor nested inside another -- the constructor has to survive as an
    array in its own right, which needs a temporary this region does not
    create, so the backend refuses it by naming the position and
    :py:meth:`apply` reports that refusal as it does any other the backend
    raises. An implied-do constructor never reaches the backend at all: the
    PSyIR frontend does not model one, so ``[ (i, i=1,n) ]`` arrives as a
    CodeBlock and is refused with every other CodeBlock.

    **A** ``DO WHILE`` **loop is accepted** and generated as a C ``while``. It
    is never spread over the team: the loops that are have a counter and a
    step for a ``TeamVectorRange`` to divide, and a while loop states neither.

    **An unlabelled** ``EXIT`` **is accepted** and generated as a C
    ``break``, which leaves the same loop the Fortran leaves. The loop it
    leaves is never spread over the team, since a lambda cannot break the
    loop it was launched over; see :py:meth:`_parallel_loops`. An ``EXIT``
    that names the construct it leaves is not modelled by the PSyIR
    frontend, so it arrives as a CodeBlock and is refused with every other
    CodeBlock -- which is also what happens to the whole ``DO`` a named
    construct wraps.

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
    be one the ABI names, as a formal's must, and every **name** left in its
    extents must be a **kernel argument**: the launch computes its scratch
    size before it enters the region, so an extent it cannot name there
    cannot be sized. A module constant is refused as an extent for that
    reason even though the body may read one elsewhere. A scalar local needs
    no scratch and is declared in the region body as before.

    A name whose value the declaration states is not left in the extent at
    all. ``integer(kind=i_def), parameter :: nfaces = 4`` beside
    ``real(kind=r_tran), dimension(nfaces) :: v_dot_n`` is the commonest
    kernel-local shape GungHo has, and the value is in the Fortran: the
    extent is resolved to ``4`` before it is read, by the same substitution
    that already replaces such a name in the body. Only names the declaration
    has values for are replaced, so ``dimension(order+1,nfaces)`` keeps the
    formal and loses the constant.

    An extent is a declared bound written as C, not a name copied over, so it
    need not be a single symbol. ``dimension(max_length,4)`` and
    ``dimension(nlayers+1)`` are both accepted, and both a formal and a local
    are read the same way. What is required is an integer expression over
    kernel arguments and literals using ``+``, ``-``, ``*`` and ``/``.
    Division is carried rather than refused because Fortran and C++ agree
    about it -- both truncate an integer quotient toward zero -- so
    ``dimension((stencil_size + 1) / 2)`` is sized as the kernel declared it.
    What the two languages do not agree about is an allocation of negative
    size, which neither defines, so a launch whose scratch extent divides
    emits a guard that says which rule it is relying on and aborts rather
    than requesting one. A call is still refused: ``max(nlayers, 1)`` and
    ``pow(nlayers, 2)`` carry a comma, which a ``shmem_size`` argument may
    not.

    **An allocated local is a declared one written the other way round.**
    ``real(kind=r_def), allocatable, dimension(:) :: partial`` followed by
    ``allocate(partial(nlayers))`` states the same shape as
    ``dimension(nlayers)`` would, one statement later, so the ``ALLOCATE``'s
    bounds are read into the declaration and both it and the matching
    ``DEALLOCATE`` are removed. Every rule about a local array then applies
    unchanged, the bounds being subject to the same grammar as a declared
    shape. Four allocations are refused instead, each by name: one whose
    extent the launch cannot evaluate, since the scratch size is computed on
    the host from the region's own scalars and ``allocate(t(minval(map)))``
    is not one of them; one inside a loop, which is a different array on each
    trip rather than one array with a size; one carrying ``stat``, ``errmsg``,
    ``source`` or ``mold``, each of which says something the reserved scratch
    does not carry; and a second allocation of an array already allocated,
    scratch being reserved once with one shape for the whole region.

    An assumed-shape local -- one carrying ``dimension(:)`` and no
    ``ALLOCATE`` -- states no shape anywhere the region can read, and is
    refused by name; a *formal* declared the same way is measured at the call
    instead, below. A shape the C writer cannot render at all, such as
    ``dimension(MAX(nlayers-n,1))``, is refused with the writer's own reason
    attached. Every one of these is refused by :py:meth:`validate` rather
    than discovered by :py:meth:`apply`.

    **A declared lower bound need not be 1.** ``dimension(0:nlayers-1)`` and
    ``dimension(-nlayers:nlayers)`` are accepted alongside
    ``dimension(nlayers)``: the View is sized by the span the declaration
    gives, ``ub - lb + 1``, and every subscript of that array has ``lb``
    subtracted from it on the way to the zero-based element, so
    ``u_e(k)`` over a local declared ``dimension(0:nlayers)`` becomes
    ``u_e((k - 0))``. The lower bound must satisfy the same grammar as the
    upper -- an integer expression over kernel arguments and literals using
    ``+``, ``-``, ``*`` and ``/`` -- and is refused on the same terms when it
    does not. A bound of 1 renders exactly the source it rendered before this
    was accepted, the span folding back to the upper bound alone.

    The subtraction is written out even where it is zero, because it is
    applied in one place -- the back-end's generation of an array accessor --
    and a subscript that escaped it would compile and give a wrong answer
    rather than fail. What is gained by omitting it is what the C++ optimiser
    removes anyway.

    **An assumed-shape formal is measured rather than refused.** A
    boundary-condition kernel writes ``real(kind=r_def), intent(in) ::
    normals(:,:)`` and lets Fortran take the extent from the actual. That
    extent is nowhere in the kernel, but it is not unknown: the PSy layer
    holds the array it is the shape of. So the region carries one integer
    formal per dimension the declaration left out, appended after the
    kernel's own and passed as ``SIZE(actual, dim=n)``, and the declaration
    is rewritten over it before anything else reads the shape. The View is
    sized by that formal, and every ``SIZE``, ``LBOUND`` and ``UBOUND`` the
    body asks of the array resolves to it exactly as a declared extent does.

    The origin, however, is not taken from the call. Fortran gives an
    assumed-shape dummy a lower bound of 1 whatever the actual was declared
    from, the dummy being a new descriptor over the actual's elements rather
    than the actual itself, so the shift of the paragraph above is the
    declaration's own 1 and the caller contributes only the size.

    A formal that states one bound and leaves the other -- ``dimension(0:)``
    -- is refused by name for that reason: its origin would come from the
    kernel and its extent from the caller, and a View shaped out of two
    places at once is what reading one declaration once exists to prevent.
    The refusal says which of the shapeless declarations it found, so a
    reader knows whether to look at the kernel or at its caller.

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
    barrier before the next statement reads it. An array constructor
    assignment is such a write even though it names its target without
    subscripting it, and all of the element assignments it becomes go inside
    one ``Kokkos::single``, which is both cheaper than one region each and
    what the Fortran said. A barrier also follows each
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

    **A name Fortran allows and C++ reserves is refused, formal or local.**
    Fortran reserves no words, so ``const``, ``new`` and ``operator`` are
    ordinary variable names -- and ``const`` is one
    ``project_eliminated_theta_q32_kernel_mod`` declares. Written out as an
    identifier it gives ``const float const,`` in the signature, which no
    compiler takes. The names are :py:attr:`_CXX_KEYWORDS` and the comparison
    is case-sensitive, only a name already lower case colliding with the
    keyword. A rename is not attempted: it would have to reach every place the
    backend writes a name, and a refusal that says which name it was costs a
    region rather than risking a wrong one.

    **A name the body reads that is not one of its arguments reaches the
    region one of four ways.** A module-level ``parameter`` declared beside
    the kernel -- ``integer(kind=i_def), parameter :: nfaces = 4`` -- is
    written into the region as its value. It has to be: a kernel module is
    ``private`` by default and publishes only its ``_code`` routine, so
    importing the name into the PSy layer would not compile. The value need
    not be a literal: one written as an arithmetic over other parameters is
    folded to what those state, and every name in it goes with it.

    An array ``parameter`` has no single value to substitute -- its subscripts
    are computed where a scalar's use is not -- so its elements are declared
    in the generated launch body as a ``const`` array of its own, beside the
    body's other locals. Not at file scope, which is what a ``parameter``
    beside a kernel most resembles: a namespace-scope array is host data and
    a device compiler will not read one. One built by a call, ``reshape``
    among them, states no elements to declare and is refused, as is one of
    rank above one, a C array being written in the one storage order that
    agrees with Fortran's subscripts only in one dimension.

    A constant *imported* from another module is passed by value, which needs
    its kind, and so needs the source of its container on PSyclone's module
    search path; without that it is refused with a message naming the module
    to add rather than a guess at its width.

    A **variable** of the kernel's own module -- ``real(kind=r_def), public ::
    profile_heights(100)``, which is how the profile reaches
    ``profile_interp_kernel_mod`` -- is state rather than a value, so the
    region takes it as a formal of its own and the PSy layer imports it from
    the kernel module to pass it: a scalar by value, an array as a read-only
    View of the extents the declaration states. What the module holds when the
    launch is made is what the region sees, which is why a body that assigns
    to one is refused rather than quietly losing the assignment. So is one the
    module keeps ``private``, since there is no name to import; one whose
    declared extents name a size the region has no argument for; and one with
    no declared extents at all, ``allocatable`` among them. A variable
    declared inside the *routine* with an initialiser is refused for a
    different reason: Fortran gives that one the ``SAVE`` attribute, so it
    keeps a value from one call to the next that the region has nowhere to
    put, and nothing outside the routine declares it for the PSy layer to
    pass.

    Last, a name appearing only as an intrinsic's ``kind`` argument -- the
    ``r_def`` of ``real(x, r_def)`` -- is none of these: it names a type, the
    cast consumes it, and the region carries the width rather than the name.

    **An intrinsic the backend cannot spell is refused by** :py:meth:`validate`
    **rather than discovered by** :py:meth:`apply`. The only thing that knows
    what the backend can write is the backend, so ``validate`` asks it: every
    intrinsic the body holds is put to the writer with its arguments replaced
    by references of their own types, and the ones that raise are reported as
    ``NAME/arity``. A second list kept beside the writer would be a list to
    keep in step.

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

    #: The option choosing between the two answers to a write two cells of
    #: one launch share. ``True`` generates a ``Kokkos::atomic_*`` update for
    #: every read-modify-write of a shared field; ``False`` generates none
    #: and requires the loop to have been coloured first, so that the cells
    #: meeting at a dof are in different launches. Absent, the choice follows
    #: the loop: a coloured loop takes the coloured arm and every other loop
    #: takes atomics, which is what makes atomics the default and makes every
    #: capture predating this option generate the source it generated then.
    #:
    #: The two are alternatives rather than a ranking. Both are correct, and
    #: which is faster is a measurement neither this class nor the branch
    #: that added it has made.
    _ATOMICS_OPTION = "atomics"

    @classmethod
    def _uses_atomics(cls, node, options):
        """Say which of the two answers to a shared write is in force.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: the transformation options.
        :type options: Optional[Dict[str, Any]]

        :returns: whether a shared update is to be generated as an atomic.
        :rtype: bool
        """
        requested = (options or {}).get(cls._ATOMICS_OPTION)
        if requested is None:
            return node.loop_type != cls._COLOURED_LOOP_TYPE
        return bool(requested)

    def _validate_atomics_option(self, node, options):
        """Check the ``"atomics"`` option against the loop it is given with.

        Each arm answers a shared write on its own, and the two together
        answer it twice: an atomic on data colouring has already made private
        to one launch costs an instruction and buys nothing. Asking for both
        is therefore a contradiction in what the caller stated rather than a
        preference to be resolved quietly, and so is asking for neither on a
        loop that has a shared write and has not been coloured -- which would
        generate a race.

        :param node: the loop that is to be captured.
        :type node: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param options: the transformation options.
        :type options: Optional[Dict[str, Any]]

        :raises TransformationError: if the option is neither absent nor a
            bool; if it is ``True`` on a coloured loop; or if it is ``False``
            on an uncoloured loop whose kernel has a shared write.
        """
        requested = (options or {}).get(self._ATOMICS_OPTION)
        if requested is not None and not isinstance(requested, bool):
            raise TransformationError(
                f"LFRicKokkosTrans' '{self._ATOMICS_OPTION}' option must be "
                f"absent or a bool, but found '{requested}'.")
        coloured = node.loop_type == self._COLOURED_LOOP_TYPE
        if requested and coloured:
            raise TransformationError(
                f"LFRicKokkosTrans' '{self._ATOMICS_OPTION}' option is True "
                "on a coloured loop. Colouring has already made every write "
                "the launch's own, so an atomic would guard data no other "
                "cell of the launch reaches; ask for one answer to a shared "
                "write or the other.")
        if (requested is False and not coloured
                and self._shared_arguments(node.kernels()[0])):
            raise TransformationError(
                f"LFRicKokkosTrans' '{self._ATOMICS_OPTION}' option is False "
                "on a loop that is not coloured, whose kernel writes a field "
                "two cells share. Colour the loop first, or leave the option "
                "out and take the atomic update.")

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
            two read here are ``"team_size"`` and ``"atomics"``; see
            :py:meth:`apply`.
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
        :raises TransformationError: if the ``"atomics"`` option and the
            loop's colouring contradict each other, as
            :py:meth:`_validate_atomics_option` states.
        """
        if not isinstance(node, LFRicLoop):
            raise TransformationError(
                "LFRicKokkosTrans expects an LFRicLoop but found "
                f"'{type(node).__name__}'.")

        self._validate_atomics_option(node, options)
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
        # Every rule below is asked of the body inlining leaves rather than
        # the one the kernel file holds: a callee brings its own loops,
        # locals, sections and constants in with it, and a rule asked before
        # the rewrite would be answering about a body that never reaches the
        # backend. The rewrite is made over a copy of the whole file, because
        # validate() must leave the schedule as it found it and because a
        # detached schedule has no Container for the callee to be found in.
        schedule = self._inlined_copy(self._schedule(kernel))
        self._validate_body(schedule)
        self._validate_sections(schedule)
        # The formals are judged with every assumed shape already measured,
        # because that is the shape apply() describes; on the copy taken
        # above, and so not on the kernel.
        self._resolve_assumed_shapes(schedule)
        self._validate_bounds(schedule)
        self._validate_formals(schedule)
        # Which names the launch reserves depends on which launch is selected,
        # and that is decided by the loops apply() will spread over the team.
        # Those are asked of a copy carrying the rewrites apply() makes,
        # because the lowering is itself a producer of loops: a kernel whose
        # only parallelisable loop is the one a section lowers to has none at
        # all until the copy is lowered. The two predicates above have already
        # shown that both rewrites succeed on this schedule. The copy is taken
        # the way _inlined_copy takes its own, and for the same reason: a
        # Routine copied alone loses the module scope its names resolve in.
        probe = self._rooted_copy(schedule)
        self._lower_allocations(probe)
        self._lower_sections(probe)
        self._substitute_bounds(probe)
        # The locals are judged on the probe rather than on the schedule,
        # because the allocation tier is what gives an allocated local its
        # shape: on the schedule it still has the deferred one the
        # declaration carried.
        self._validate_locals(probe, self._parallel_loops(probe))
        self._validate_intrinsics(probe)
        # On the probe, and after the lowering, because the shape of an
        # update is what decides whether an atomic can carry it out and the
        # lowering is what settles that shape.
        if self._uses_atomics(node, options):
            self._validate_shared_updates(kernel, probe)
        self._constants(schedule)
        # The file-scope constants are described here as well as in apply(),
        # so that an array parameter the generated unit could not declare is
        # a refusal rather than a failure part-way through the capture.
        self._constant_arrays(schedule)

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
        self._inline_calls(schedule)
        self._lower_allocations(schedule)
        self._lower_sections(schedule)
        self._substitute_bounds(schedule)
        self._substitute_constants(schedule)
        region, actuals, constants = self._region(
            kernel, node, schedule, options)
        try:
            cpp = KokkosWriter()(region)
        except (VisitorError, ValueError, TypeError) as err:
            raise TransformationError(
                f"LFRicKokkosTrans cannot express '{kernel.name}' in the "
                f"Kokkos backend: {err}") from err
        self._call_region(node, region, actuals, constants)
        return cpp


__all__ = ["LFRicKokkosTrans"]
