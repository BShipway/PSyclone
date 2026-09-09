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
rules ``_validate_kernel_metadata`` bundles are each askable on their own:
:py:meth:`LFRicKokkosContractMixin._validate_evaluator` and
:py:meth:`LFRicKokkosContractMixin._validate_field_types`. The bundling method
calls them rather than repeating them, and the second shares the per-argument
:py:meth:`LFRicKokkosContractMixin._validate_field_type` with the argument walk
in ``_validate_kernel_metadata``, so the order the bundled refusals come in is
unchanged by their being nameable apart.

Some predicates of that set are not here. ``_validate_bounds`` lives beside
the declaration reading it predicts, in ``LFRicKokkosBoundsMixin``; the rules
about where the loop iterates -- ``_validate_iteration_space``,
``_validate_halo_depth`` and the ``_validate_loop`` that bundles them -- are
``LFRicKokkosIterationMixin``'s; and the rule about how the loop may write to
what it shares -- ``_validate_shared_updates`` -- is
``LFRicKokkosWriteMixin``'s. Each is asked of ``LFRicKokkosTrans`` exactly as
these are.

The sibling mixins are reached through ``cls``, resolved on
``LFRicKokkosTrans``: :py:meth:`LFRicKokkosContractMixin._validate_sections`
predicts ``cls._lower_sections`` over a copy, and
:py:meth:`LFRicKokkosContractMixin._validate_formals` and
:py:meth:`LFRicKokkosContractMixin._validate_locals` ask ``cls._c_type``,
``cls._extent_names`` and ``cls._CELL_COUNT``, and
:py:meth:`LFRicKokkosContractMixin._validate_field_type` asks
``cls._view_intrinsics``. Calling a method here directly on this mixin is
therefore not supported.
"""

from psyclone.core import AccessType
from psyclone.psyir.nodes import (
    ArrayConstructor, ArrayReference, Assignment, Call, CodeBlock,
    IntrinsicCall, Range, Reference)
from psyclone.psyir.nodes.array_mixin import ArrayMixin
from psyclone.psyir.symbols import ArrayType, DataSymbol
from psyclone.psyir.transformations import TransformationError


class LFRicKokkosContractMixin:
    """The capture contract, as a set of side-effect-free predicates.

    Every question here is whether a loop, a kernel or a schedule is inside
    what the generated region can express. What a symbol is in C terms is
    ``LFRicKokkosTypesMixin``; what its declaration says its shape is, and
    the one predicate that asks, is ``LFRicKokkosBoundsMixin``; how the
    region's arguments are built is ``LFRicKokkosArgumentMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: Accesses a cell-parallel launch can honour. ``READ`` and ``READWRITE``
    #: are each cell's own and need nothing -- LFRic's metadata admits
    #: ``gh_readwrite`` on a discontinuous space only; ``WRITE`` is each
    #: cell's own on a discontinuous space too, but on a continuous one the
    #: cells meeting at a dof each *replace* its value; ``INC`` and
    #: ``READINC`` are shared between those cells and each *contribute* to
    #: it. Both kinds of sharing are answered either by an atomic -- a store
    #: for a replacement, an update for a contribution -- or by colouring the
    #: loop, which is the choice ``LFRicKokkosTrans._uses_atomics`` makes.
    #: Which of the two is in force is not asked here: both make the same
    #: accesses safe.
    #: ``REDUCTION`` is not here: it is shared between *all* cells
    #: rather than between neighbours, so neither answer reaches it.
    _SAFE_ACCESSES = (AccessType.READ, AccessType.WRITE, AccessType.READWRITE,
                      AccessType.INC, AccessType.READINC)
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
    #: The bound names in
    #: :py:attr:`LFRicKokkosIterationMixin._BOUND_NAMES` are declared by one
    #: shape or another and are always checked.
    _GENERATED_NAMES = ("body", "league_size", "probe", "rank",
                        "scratch_bytes", "team", "team_size")

    #: Stencil shapes the region's per-cell arguments describe. A 2-D stencil
    #: hands the kernel a sliced dofmap and a sliced size array, both of them
    #: array formals, so both become Views with the cell index appended --
    #: which is what ``map_w3(:,cell)`` already does and needs nothing new.
    #:
    #: A 1-D or region stencil hands the size to the kernel as a *scalar*
    #: formal, fed from ``x_stencil_size(cell)``. That is one cell's value,
    #: and a region runs every cell at once, so it crosses the ABI as the
    #: whole rank-1 array and is subscripted by the launch's own cell; see
    #: :py:meth:`LFRicKokkosArgumentMixin._per_cell_scalars`. The dofmap
    #: those two shapes hand over declares that per-cell size as its own
    #: second extent, which no View can be strided by, so the region carries
    #: the dofmap's storage extent beside it.
    #:
    #: ``xory1d`` is absent. It carries a direction argument on top of the
    #: 1-D size, chosen per cell in the algorithm layer, and nothing here
    #: describes it. It is refused by name.
    _SUPPORTED_STENCILS = ("cross", "cross2d", "region")

    #: Evaluator shapes whose basis data the region already carries. XYoZ
    #: quadrature adds two point counts, two weight arrays and one basis
    #: array per function space that asks for one, shaped
    #: ``(dim, ndf, np_xy, np_z)``. Face quadrature adds a face count, one
    #: point count, a *rank-2* weight array over the two of them and a basis
    #: array shaped ``(dim, ndf, np_xyz, nfaces)``. An evaluator adds no rule
    #: of its own at all: it tabulates the basis at the nodal points of a
    #: target function space, giving ``(dim, ndf, ndf of the target)`` and no
    #: weights. Every one of those is an argument the PSy layer has computed
    #: before the loop and every extent of it is a formal of the same kernel,
    #: so the existing scalar and View descriptions cover them whole.
    #:
    #: Edge quadrature is absent. It carries an edge count where face
    #: quadrature carries a face count and is otherwise the same shape of
    #: argument, but the released model has no kernel asking for it, so
    #: nothing here has been measured against the model for it. It is refused
    #: by name rather than accepted on the strength of the resemblance.
    _SUPPORTED_SHAPES = (
        "gh_quadrature_xyoz", "gh_quadrature_face", "gh_evaluator")

    #: Names a kernel symbol may not carry into the generated region. Fortran
    #: and C++ do not reserve the same words, so a perfectly ordinary Fortran
    #: dummy argument or local -- ``const``, ``operator``, ``class``, ``new``
    #: -- becomes a syntax error the moment the backend writes it out as an
    #: identifier. The region is refused rather than the name rewritten: a
    #: rename would have to reach every place the backend writes a name, and
    #: nothing here can promise that today.
    #:
    #: Fortran is case-insensitive and PSyIR holds these names as the source
    #: wrote them, so the comparison is case-sensitive on purpose: only a name
    #: that is *already* lower case collides with the C++ keyword, and a
    #: kernel writing ``CONST`` would generate valid C++.
    #:
    #: The alternative tokens -- ``and``, ``or``, ``not`` and the rest -- are
    #: keywords in C++ as much as ``if`` is, and are as likely as any to be a
    #: Fortran variable, so they are listed with the others.
    _CXX_KEYWORDS = frozenset((
        "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand",
        "bitor", "bool", "break", "case", "catch", "char", "char8_t",
        "char16_t", "char32_t", "class", "compl", "concept", "const",
        "consteval", "constexpr", "constinit", "const_cast", "continue",
        "co_await", "co_return", "co_yield", "decltype", "default", "delete",
        "do", "double", "dynamic_cast", "else", "enum", "explicit", "export",
        "extern", "false", "float", "for", "friend", "goto", "if", "inline",
        "int", "long", "mutable", "namespace", "new", "noexcept", "not",
        "not_eq", "nullptr", "operator", "or", "or_eq", "private",
        "protected", "public", "register", "reinterpret_cast", "requires",
        "return", "short", "signed", "sizeof", "static", "static_assert",
        "static_cast", "struct", "switch", "template", "this", "thread_local",
        "throw", "true", "try", "typedef", "typeid", "typename", "union",
        "unsigned", "using", "virtual", "void", "volatile", "wchar_t",
        "while", "xor", "xor_eq"))

    @classmethod
    def _validate_cxx_name(cls, symbol, description):
        """Check that one kernel symbol's name is writable as C++.

        :param symbol: the kernel symbol the region would name.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param str description: what the symbol is to the kernel, for the
            refusal to say: ``"formal"`` or ``"local"``.

        :raises TransformationError: if the name is a C++ keyword, which the
            generated region could not write as an identifier.
        """
        if symbol.name in cls._CXX_KEYWORDS:
            raise TransformationError(
                f"LFRicKokkosTrans cannot name the kernel {description} "
                f"'{symbol.name}' in the generated region, because it is a "
                "C++ keyword.")

    @classmethod
    def _validate_evaluator(cls, kernel):
        """Check that every basis shape the kernel asks for is modelled.

        A kernel may name more than one shape, in which case each function
        space it declares carries a basis array per shape. They are checked
        one at a time so that the refusal names the shape that is not
        modelled rather than the whole set.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :raises TransformationError: if the kernel asks for a basis shape
            outside :py:attr:`_SUPPORTED_SHAPES`.
        """
        for shape in kernel.eval_shapes:
            if shape not in cls._SUPPORTED_SHAPES:
                raise TransformationError(
                    f"LFRicKokkosTrans does not support the '{shape}' "
                    f"evaluator shape.")

    @classmethod
    def _validate_field_type(cls, argument):
        """Check one argument's intrinsic type, if it is a field.

        Anything that is not a field is passed over rather than refused, so
        that the rule can be walked over a whole argument list.

        A field's data reaches the region as the elements of a View, so what
        it may be is what ``cls._view_intrinsics`` says a View's elements may
        be rather than a list kept here. LFRic's
        :py:attr:`~psyclone.domain.lfric.LFRicConstants.\
VALID_FIELD_DATA_TYPES` admits ``gh_real`` and ``gh_integer`` and no third
        intrinsic, so no kernel in the model reaches this refusal; it is kept
        because what it states is a property of the ABI rather than of
        today's metadata, and a field intrinsic LFRic added would otherwise
        reach the backend as a type it has no row for.

        The kind is a separate question, asked of the kernel's own
        declaration by :py:meth:`_validate_formals`: an integer field
        declared at a width the ABI carries no C type for is refused there,
        naming the kind.

        :param argument: the kernel argument to check.
        :type argument: :py:class:`psyclone.lfric.LFRicKernelArgument`

        :raises TransformationError: if the argument is a field whose data is
            of an intrinsic no View can hold.
        """
        if argument.argument_type != "gh_field":
            return
        intrinsics = list(cls._view_intrinsics())
        if argument.intrinsic_type not in intrinsics:
            phrase = " and ".join(
                [", ".join(intrinsics[:-1]), intrinsics[-1]])
            raise TransformationError(
                f"LFRicKokkosTrans supports only {phrase} fields, but "
                f"'{argument.name}' is {argument.intrinsic_type}.")

    @classmethod
    def _validate_field_types(cls, kernel):
        """Check every field the kernel takes for an intrinsic a View holds.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :raises TransformationError: if any field argument's data is of an
            intrinsic no View can hold, for the reason
            :py:meth:`_validate_field_type` gives.
        """
        for argument in kernel.arguments.args:
            cls._validate_field_type(argument)

    @classmethod
    def _validate_kernel_metadata(cls, kernel):
        """Check the LFRic metadata of the kernel to be captured.

        The order the refusals come in is part of the contract: the checks
        with no argument of their own run first, and then the argument list
        is walked once, each argument answering every rule before the next
        argument is looked at. A kernel with two blockers on two arguments
        therefore reports the first *argument's*, which is not what running
        the separately askable rules in turn would report.
        :py:meth:`_validate_field_types` shares
        :py:meth:`_validate_field_type` with the walk below rather than
        restating it, so there is one copy of the rule and two ways to ask
        it.

        :param kernel: the kernel the loop holds.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

        :raises TransformationError: if the kernel needs quadrature or
            evaluator data, is a CMA kernel, takes an argument that is not a
            field, a scalar or an LMA operator, takes an access a
            cell-parallel launch cannot honour, takes a field of an intrinsic
            no View can hold, or uses a stencil shape outside
            :py:attr:`_SUPPORTED_STENCILS`.
        """
        cls._validate_evaluator(kernel)
        if kernel.cma_operation is not None:
            raise TransformationError(
                "LFRicKokkosTrans does not support CMA operators.")
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
                    "answers a write shared between neighbouring cells -- "
                    "gh_inc, gh_readinc, and gh_write to a continuous space "
                    "-- with an atomic or with colouring, and neither of "
                    "those answers a value shared between every cell.")
            if argument.argument_type != "gh_field":
                continue
            cls._validate_field_type(argument)
            if argument.stencil:
                shape = str(argument.stencil.name).lower()
                if shape not in cls._SUPPORTED_STENCILS:
                    raise TransformationError(
                        f"LFRicKokkosTrans supports the "
                        f"{', '.join(cls._SUPPORTED_STENCILS)} stencil shapes "
                        f"only, but '{argument.name}' has '{shape}'.")

    @staticmethod
    def _validate_body(schedule):
        """Check that nothing in the body escapes the generated region.

        A call is not asked about here. The generated region still has no
        Fortran to call into, but by the time this runs the body has been
        through
        :py:meth:`LFRicKokkosInlineMixin._inline_calls`, which has either made
        every callee's statements the kernel's own or refused the capture
        naming the call it could not. A second rule here would only be able
        to report a call that rewrite had already accepted.

        :param schedule: the kernel schedule being captured, after inlining.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if the body holds a
            :py:class:`~psyclone.psyir.nodes.CodeBlock`, which the generated
            region has no way to express.
        """
        if schedule.walk(CodeBlock):
            raise TransformationError(
                "LFRicKokkosTrans cannot capture a CodeBlock.")

    @classmethod
    def _is_array_valued(cls, assignment):
        """Answer whether ``assignment`` is one that lowering rewrites.

        The two shapes it answers for are a written section, ``a(2:n) = 0.0``,
        and a whole array named with no accessor at all, ``pv_at_quad = 0.0``,
        which is the same statement written the shorter way. An array
        constructor on the right, ``cells(:) = [2, 3, 4, 5]``, is excluded
        even though it is a section: its values are positional and
        :py:class:`~psyclone.psyir.backend.c.CWriter` renders it element by
        element, so lowering would take a statement the backend can already
        write and leave a subscripted constructor in its place. An
        array-valued intrinsic on the right is excluded for the same reason
        and is
        :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_intrinsic_mixin.LFRicKokkosIntrinsicMixin._written_as_a_nest`'s
        to say so.

        :param assignment: the assignment to judge.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`

        :returns: whether :py:meth:`LFRicKokkosTrans._lower_sections` rewrites
            this assignment.
        :rtype: bool
        """
        if isinstance(assignment.rhs, ArrayConstructor):
            return False
        if cls._written_as_a_nest(assignment):
            return False
        if assignment.walk(Range):
            return True
        target = assignment.lhs
        return (not isinstance(target, ArrayMixin)
                and isinstance(target.symbol, DataSymbol)
                and isinstance(target.symbol.datatype, ArrayType))

    @classmethod
    def _validate_sections(cls, schedule):
        """Check that every array-valued assignment can be lowered to a loop.

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

        A section that is not part of an assignment is out of reach of
        lowering, but it is only this rule's to refuse when it stands on its
        own. Passed to a routine -- ``call convert(field(:,k))`` -- it is the
        call that decides: inlining the callee takes the argument away with
        it, and where the callee cannot be inlined that refusal is the
        blocker, named by
        :py:meth:`LFRicKokkosInlineMixin._inline_calls`. So this rule steps
        aside rather than reporting the argument as a second, weaker reason
        for the same refusal. The shape an ``ALLOCATE``
        states is not a section at all: it is a declaration written as a
        statement, and :py:meth:`_lower_allocations` is what reads it.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if a section stands outside an assignment
            other than as the argument of a call, and so is beyond what
            lowering can reach.
        :raises TransformationError: if an array-valued assignment cannot be
            lowered, in which case the reason is the one PSyclone gives.
        """
        for section in schedule.walk(Range):
            if section.ancestor(Assignment) is not None:
                continue
            call = section.ancestor(Call)
            if isinstance(call, IntrinsicCall) and \
                    call.intrinsic in cls._ALLOCATIONS:
                continue
            if call is not None and not isinstance(call, IntrinsicCall):
                continue
            raise TransformationError(
                "LFRicKokkosTrans cannot capture an array section outside "
                "an assignment: only a whole-column assignment can be "
                "lowered to a loop the generated region can express.")
        if not any(cls._is_array_valued(assignment)
                   for assignment in schedule.walk(Assignment)):
            return
        try:
            cls._lower_sections(schedule.copy())
        except TransformationError as err:
            raise TransformationError(
                "LFRicKokkosTrans cannot lower an array section to a "
                f"loop: {err}") from err

    @classmethod
    def _validate_formals(cls, schedule):
        """Check that every kernel formal has a place on the C ABI.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if the kernel already declares one of
            the names the generated signature adds for its own bounds; see
            :py:attr:`LFRicKokkosIterationMixin._BOUND_NAMES`.
        :raises TransformationError: if a formal's name is a C++ keyword; see
            :py:attr:`_CXX_KEYWORDS`.
        :raises TransformationError: if a formal's kind is not one
            :py:attr:`_C_TYPES` maps, or if an array formal's extent is not
            itself a formal, so the generated View could not be sized.
        """
        table = schedule.symbol_table
        formals = table.argument_list
        names = {symbol.name for symbol in formals}
        # Sorted so that a kernel colliding with two of them names the same
        # one on every run.
        for bound in sorted(names.intersection(cls._BOUND_NAMES)):
            raise TransformationError(
                f"LFRicKokkosTrans adds '{bound}' to the generated "
                "signature, but the kernel already declares it.")
        for symbol in formals:
            cls._validate_cxx_name(symbol, "formal")
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

        :raises TransformationError: if the kernel declares a local named in
            :py:attr:`LFRicKokkosIterationMixin._BOUND_NAMES`, which every
            launch shape may declare.
        :raises TransformationError: if the kernel selects a team launch --
            by having an automatic array, or a loop to spread over the team --
            and declares a local named in :py:attr:`_GENERATED_NAMES`, which
            that launch declares.
        :raises TransformationError: if a local's name is a C++ keyword; see
            :py:attr:`_CXX_KEYWORDS`.
        :raises TransformationError: if a local array's kind is not one
            :py:attr:`_C_TYPES` maps, or if one of its extents is not a
            kernel argument, so the scratch View could not be sized.
        """
        table = schedule.symbol_table
        names = {symbol.name for symbol in table.argument_list}

        locals_ = list(table.automatic_datasymbols)
        generated = set(cls._BOUND_NAMES)
        if any(symbol.is_array for symbol in locals_) or bool(parallel_loops):
            generated.update(cls._GENERATED_NAMES)
        # Sorted so that a kernel colliding with two of them names the same
        # one on every run.
        for name in sorted(generated.intersection(
                symbol.name for symbol in locals_)):
            raise TransformationError(
                f"LFRicKokkosTrans' generated launch declares '{name}', but "
                "the kernel declares a local of that name.")

        # Sorted for the reason the collision above is: a kernel with two
        # such locals names the same one on every run.
        for symbol in sorted(locals_, key=lambda symbol: symbol.name):
            cls._validate_cxx_name(symbol, "local")

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
    def _cell_position(cls, kernel, loop, formals, actuals):
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

        A colouring makes the actual ``cmap(colour, cell)`` rather than
        ``cell``, and that is the same value by a longer route: the region
        already declares its own cell from the same lookup, so the position
        it derives from it is the cell of the mesh either way.

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
            first actual is neither a reference to the loop's own variable
            nor that variable's entry in the colour map.
        """
        if not kernel.arguments.has_operator():
            return None
        actual = actuals[0]
        if not cls._is_the_loops_cell(kernel, loop, actual):
            raise TransformationError(
                f"LFRicKokkosTrans expected the PSy layer to supply the "
                f"loop's own cell index as the first argument of "
                f"'{kernel.name}', but found '{actual.debug_string()}'.")
        return formals[0].name

    @classmethod
    def _is_the_loops_cell(cls, kernel, loop, actual):
        """Say whether an actual is the cell the loop is on.

        Two spellings mean it, and no third one does. Uncoloured, LFRic
        passes the loop's own variable. Coloured, it passes that variable's
        entry in the kernel's colour map, with the colour the outer Fortran
        loop is on: ``cmap(colour, cell)``. The map and the index are checked
        against the symbols the schedule holds, so an unrelated array
        reference in the first position is refused rather than mistaken for
        a cell.

        :param kernel: the kernel being captured.
        :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`
        :param loop: the loop the kernel sits in.
        :type loop: :py:class:`psyclone.domain.lfric.LFRicLoop`
        :param actual: the first actual argument the PSy layer supplies.
        :type actual: :py:class:`psyclone.psyir.nodes.Node`

        :returns: whether that actual is this loop's cell index.
        :rtype: bool
        """
        if not isinstance(actual, Reference):
            return False
        if not isinstance(actual, ArrayReference):
            return actual.symbol is loop.variable
        if loop.loop_type != cls._COLOURED_LOOP_TYPE:
            return False
        indices = actual.indices
        return (actual.symbol is kernel.colourmap
                and len(indices) == 2
                and isinstance(indices[1], Reference)
                and indices[1].symbol is loop.variable)


__all__ = ["LFRicKokkosContractMixin"]
