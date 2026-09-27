# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""A deliberately small Kokkos backend for whole LFRic loop regions.

What a region is made of is described in
:py:mod:`~psyclone.psyir.backend.kokkos_region` and re-exported here, because
this module's name is the one every caller and every test imports those
records by and moving them was not meant to move that.
"""

from dataclasses import replace

from psyclone.psyir.backend.c import CWriter
from psyclone.psyir.backend.c_intrinsics_mixin import (
    INTEGER_INTRINSIC_ALTERNATIVES)
from psyclone.psyir.backend.kokkos_array_expression import KokkosScratch
from psyclone.psyir.backend.kokkos_array_expression_mixin import (
    KokkosArrayExpressionMixin)
from psyclone.psyir.backend.kokkos_intrinsics_mixin import (
    KokkosIntrinsicsMixin)
from psyclone.psyir.backend.kokkos_constant import KokkosConstant
from psyclone.psyir.backend.kokkos_staging import (
    include_line, staging_epilogue, view_declaration)
from psyclone.psyir.backend.kokkos_region import (
    KokkosAlias, KokkosColourMap, KokkosRegion, KokkosScalar, KokkosView,
    extent_names, is_extent, is_offset)
from psyclone.psyir.backend.kokkos_spread_extent import spread_extents
from psyclone.psyir.backend.kokkos_team_scalars import (
    team_private_scalars)
from psyclone.psyir.backend.kokkos_launch import (
    hierarchical_launch, member_local_declaration, member_local_definition,
    range_launch, scratch_probe_definition, team_launch, team_scratch_items)
from psyclone.psyir.backend.kokkos_launch_dof import dof_launch
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.backend.kokkos_validation_mixin import (
    KokkosValidationMixin)
from psyclone.psyir.nodes import (
    ArrayReference, Assignment, Literal, Range, Reference)


class KokkosWriter(KokkosValidationMixin, KokkosIntrinsicsMixin,
                   KokkosArrayExpressionMixin, CWriter):
    """Generate a C++/Kokkos translation unit for a captured region.

    The intrinsics and the handling of array-valued expressions are inherited
    rather than written here, from
    :py:mod:`~psyclone.psyir.backend.kokkos_intrinsics_mixin` and
    :py:mod:`~psyclone.psyir.backend.kokkos_array_expression_mixin`, the
    lowering the second of them drives being
    :py:mod:`~psyclone.psyir.backend.kokkos_array_expression`. Both mixins
    precede :py:class:`~psyclone.psyir.backend.c.CWriter` in the bases so that
    their handlers are found first and fall through to the C writer's by
    ``super()``. The checks made of a region before anything is generated
    are :py:mod:`~psyclone.psyir.backend.kokkos_validation_mixin`'s.
    """

    #: The C types this writer will declare. ``bool`` is the one with no
    #: width behind it: the driving transformation puts a Fortran ``logical``
    #: on the ABI by conversion rather than by matching kinds, so nothing here
    #: has to know what ``l_def`` measures.
    _SUPPORTED_TYPES = ("bool", "double", "float", "int")

    def __init__(self, **kwargs):
        """Create a writer holding no region.

        :param kwargs: additional keyword arguments for
            :py:class:`~psyclone.psyir.backend.c.CWriter`.
        :type kwargs: unwrapped dict
        """
        super().__init__(**kwargs)
        self._views = {}
        self._kind_types = {}
        self._parallel_loops = ()
        # How many of the region's chosen loops the visitor is currently
        # inside. A team-level array write is one made at depth zero; inside a
        # chosen loop the members already have disjoint iterations, so the
        # write is theirs alone and needs no ``Kokkos::single``.
        self._parallel_depth = 0
        # The scalars each chosen loop declares inside its own lambda, keyed
        # by the ``id`` of the loop, from
        # :py:func:`~psyclone.psyir.backend.kokkos_team_scalars.team_private_scalars`.
        self._private_scalars = {}

    def __call__(self, region: KokkosRegion) -> str:
        """Generate code for ``region``.

        The region's :py:attr:`KokkosRegion.kind_types` are in force for the
        duration of the call and cleared afterwards, so that a writer reused
        for a second region does not carry the first one's widths into it.

        One of four launch shapes is generated. A region whose
        :py:attr:`KokkosRegion.dof` is set takes the dof launch, ahead of
        every other selection because a dof region has no cells to give a
        team; otherwise a region naming any
        :py:attr:`KokkosRegion.parallel_loops` takes the hierarchical launch;
        otherwise one describing any :py:attr:`KokkosRegion.scratch` takes the
        flat team launch; otherwise the range launch. None of the older three
        reads the fields it does not select on, so a region built before they
        existed is generated exactly as it was then; see
        :py:func:`~psyclone.psyir.backend.kokkos_launch.range_launch`,
        :py:func:`~psyclone.psyir.backend.kokkos_launch.team_launch`,
        :py:func:`~psyclone.psyir.backend.kokkos_launch.hierarchical_launch`
        and :py:func:`~psyclone.psyir.backend.kokkos_launch_dof.dof_launch`.

        Every scalar a chosen loop writes is declared inside that loop's
        lambda, so that each member of the team owns its own; which scalars
        those are, and which spread loops are refused because a scalar they
        write cannot be anyone's own, is decided by
        :py:func:`~psyclone.psyir.backend.kokkos_team_scalars.team_private_scalars`.

        :param region: the captured region to generate.

        :returns: a complete C++ translation unit.

        :raises TypeError: as :py:meth:`_validate` does.
        :raises ValueError: as :py:meth:`_validate` does; if the body indexes
            an array for which the region described neither a View nor
            scratch; and if a loop the region asks to spread over the team
            writes a scalar that cannot be made private to a member.
        """
        self._validate(region)
        self._private_scalars, private_names = team_private_scalars(region)
        self._views = {
            argument.name: argument for argument in region.arguments
            if isinstance(argument, KokkosView)
        }
        # Scratch joins the same table so that ``arrayreference_node`` resolves
        # ``x_new(k)`` and ``mr_v(df)`` by one lookup.
        self._views.update({item.name: item for item in region.scratch})
        self._views.update({item.name: item for item in region.constants})
        # An alias is a second name for an array already in the table, so it
        # joins the table as a copy of that array's description under its own
        # name. That is what makes ``p(k)`` subscript the View the pointer
        # was aimed at, with the origin the target's declaration carries: an
        # alias has no origin of its own to apply, and applying the wrong one
        # would compile and return the wrong answer.
        self._views.update({
            alias.name: replace(self._views[alias.targets[0]],
                                name=alias.name)
            for alias in region.aliases})
        self._kind_types = dict(region.kind_types)
        self._parallel_loops = region.parallel_loops
        self._schedule = region.schedule

        signature = ",\n    ".join(
            self._argument_declaration(argument)
            for argument in region.arguments)
        views = "\n".join(
            view_declaration(argument) for argument in region.arguments
            if isinstance(argument, KokkosView))
        # Both team shapes need the policy and its member type. Only a region
        # with scratch needs the space its Views are placed in, and the
        # hierarchical shape is reached without scratch, so that alias is
        # conditioned separately rather than riding along unused.
        team_aliases = "".join(
            f"  {alias}\n" for alias in (
                "using TeamPolicy = Kokkos::TeamPolicy<>;",
                "using TeamMember = TeamPolicy::member_type;",
            )) if (region.scratch or region.parallel_loops) and not \
            region.dof else ""
        if team_scratch_items(region):
            team_aliases += (
                "  using ScratchSpace = "
                "Kokkos::DefaultExecutionSpace::scratch_memory_space;\n")

        # The flat team shape nests the body one level deeper, inside the
        # TeamThreadRange lambda. The hierarchical one does not: its body sits
        # directly in the functor, as the range shape's does.
        scratch_names = {item.name for item in region.scratch}
        alias_names = {alias.name for alias in region.aliases}
        # A dof region's scratch is all member-local and needs no team, so
        # its body sits in the range lambda as a range region's does.
        self._depth = 3 if region.scratch and not (
            region.parallel_loops or region.dof) else 2
        # Ahead of the locals, and left after the scratch constructions each
        # launch shape emits although an ``AnonymousSpace`` handle no longer
        # needs its target to have been constructed first: moving it would
        # rewrite the body of every region that already carries one, for
        # nothing.
        alias_declarations = "".join(
            f"{self._nindent}"
            f"{self._alias_declaration(alias, self._views)}\n"
            for alias in region.aliases)
        local_declarations = alias_declarations + "".join(
            self.gen_local_variable(symbol)
            for symbol in region.schedule.symbol_table.automatic_datasymbols
            # A scratch array is declared as a View over team scratch, so its
            # symbol must not also be declared here. ``gen_declaration``
            # renders an array local as ``double * restrict x_new`` -- a
            # pointer to nothing, which compiles and would shadow the View.
            # A scalar every use of which is inside a loop spread over the
            # team is declared inside that loop's lambda instead of here, so
            # that each member owns one. A scalar the body also uses outside
            # those loops keeps this declaration and is shadowed by the
            # private one, because the outer uses still need something to
            # name; which of the two a symbol is is decided by
            # ``team_private_scalars`` and not here.
            # An alias is declared above as a handle of its target's type,
            # so its symbol must not also be declared here: the pointer's
            # PSyIR datatype is an array of deferred shape, which
            # ``gen_declaration`` renders as a pointer to nothing.
            if symbol.name not in scratch_names
            and symbol.name not in alias_names
            and symbol.name not in private_names)
        if region.cell_position is not None:
            # First, and prepended here rather than in each launch shape: all
            # three place these declarations immediately after establishing
            # their own cell index, which is the only thing this one reads.
            local_declarations = (
                f"{self._nindent}const int {region.cell_position} = "
                f"{region.cell_index} + 1;\n") + local_declarations
        if region.colour_map is not None:
            # Ahead of the line above, which reads the cell index this one
            # establishes, and prepended here for the same reason: all three
            # launch shapes want it immediately after their own index, and
            # each of them has just declared that.
            local_declarations = (
                f"{self._nindent}const int {region.cell_index} = "
                f"{region.colour_map.name}({region.colour_map.colour} - 1, "
                f"{region.colour_map.index}) - 1;\n") + local_declarations
        body = "".join(
            self._visit(child) for child in region.schedule.children)
        # Rendered here, while this writer's kinds and Views are in force,
        # because the launch that consumes it visits no PSyIR.
        extents = spread_extents(region, self._visit)
        constant_indent, self._depth = self._nindent, 0
        self._views, self._kind_types = {}, {}
        self._parallel_loops = ()
        self._private_scalars = {}

        if region.dof:
            # The cell shapes declare a member-local array among their
            # scratch constructions. The dof shape has none, so its arrays,
            # which the validation has shown are all member-local, are
            # declared with the locals, behind the constants added below.
            local_declarations = "".join(
                member_local_declaration(item, constant_indent)
                for item in region.scratch) + local_declarations

        # Inside the body, not at file scope: nvcc will not read a namespace
        # scope array from device code. First, since a constant reads nothing.
        local_declarations = "".join(
            f"{constant_indent}const {item.c_type} {item.name}"
            f"[{len(item.values)}] = "
            f"{{{', '.join(self._visit(value) for value in item.values)}}};\n"
            for item in region.constants) + local_declarations

        if region.dof:
            launch = dof_launch(region, local_declarations, body)
        elif region.parallel_loops:
            launch = hierarchical_launch(
                region, local_declarations, body, extents)
        elif region.scratch:
            launch = team_launch(region, local_declarations, body)
        else:
            launch = range_launch(region, local_declarations, body)

        unit = (
            # Ahead of the region, since a member of the team declares one
            # inside the functor and a template may not be defined there.
            f"{member_local_definition(region)}"
            f"{scratch_probe_definition(region)}"
            f'extern "C" void {region.name}(\n'
            f"    {signature}) {{\n"
            # Kokkos does not treat an uninitialised runtime as an error: the
            # region prints a diagnostic to stderr and then runs correctly but
            # single-threaded, so the omission survives every build and every
            # answer-based test, and shows up only as lost performance. Naming
            # the region here turns that into an immediate, attributable stop.
            "  if (!Kokkos::is_initialized()) {\n"
            f'    Kokkos::abort("{region.name}: Kokkos region entered before '
            'Kokkos::initialize()");\n'
            "  }\n\n"
            "  using MemorySpace = "
            "Kokkos::DefaultExecutionSpace::memory_space;\n"
            "  using Unmanaged = "
            "Kokkos::MemoryTraits<Kokkos::Unmanaged>;\n"
            "  using ReadOnly = Kokkos::MemoryTraits<"
            "Kokkos::Unmanaged | Kokkos::RandomAccess>;\n"
            f"{team_aliases}"
            "\n"
            f"{views}\n\n"
            f"{launch}"
            "  Kokkos::fence();\n"
            f"{staging_epilogue(region)}"
            "}\n")
        return f"{self._includes(unit)}{unit}"

    @staticmethod
    def _includes(unit):
        """Return the ``#include`` lines the generated unit needs.

        ``<Kokkos_Core.hpp>`` always, and ``<algorithm>`` for a region that
        carries one of the standard-library calls the C writer spells an
        integer ``MAX`` or ``MIN`` with. Which headers are needed is read
        from the generated text rather than from the region's description,
        because the description says what the region is made of and not which
        of the writers' spellings that came out as: an integer maximum
        reaches the unit through a declared bound, a scratch size and a View
        extent alike, and each of those is a string by the time it is here.

        A region carrying none of them gets the one line it always got, byte
        for byte, since a header emitted unconditionally would rewrite every
        capture already in the model for a call it does not make.

        The staging header joins them on the same terms, and on the same
        evidence: a region that obtains its Views through it names it in the
        text, once, however many Views it has.

        :param str unit: the generated translation unit, without its
            includes.

        :returns: the include lines, ending with the blank line that
            separates them from the unit.
        :rtype: str
        """
        headers = ["<Kokkos_Core.hpp>"]
        if any(f"{call}(" in unit
               for call in INTEGER_INTRINSIC_ALTERNATIVES.values()):
            headers.append("<algorithm>")
        lines = [f"#include {header}\n" for header in headers]
        if "lfric_kokkos::" in unit:
            lines.append(f"{include_line()}\n")
        return "".join(lines) + "\n"

    def assignment_node(self, node) -> str:
        """Emit a pointer assignment as a View handle copy.

        ``p => x`` is not a statement about values: it aims a second name at
        the storage ``x`` names, so that every later read of ``p(k)`` reads
        ``x(k)``. A Kokkos ``View`` is reference-semantic, so the C++ that
        says the same thing is the handle copy ``p = x;`` -- one word of
        assignment, and nothing copied element by element.

        It is written here rather than left to the writers below because
        every one of them would say something else about it. The C writer
        emits ``p = x;`` for the names alone and would be right by accident,
        but a whole-array reference on the right of an assignment is an array
        expression, so
        :py:meth:`~psyclone.psyir.backend.kokkos_array_expression_mixin.\
KokkosArrayExpressionMixin.assignment_node` would first lower it into the
        loop nest that copies ``x`` into ``p`` element by element -- storage
        the alias does not have, and a copy the Fortran did not ask for. So
        the pointer case is answered before either of them is reached, and
        the two references are written as the names they are: no subscript,
        no origin, no lowering.

        :param node: the assignment in the captured body.
        :type node: :py:class:`psyclone.psyir.nodes.Assignment`

        :returns: the handle copy for a pointer assignment, and whatever the
            writers below make of any other assignment.
        :rtype: str
        """
        if node.is_pointer:
            if isinstance(node.rhs, ArrayReference):
                return self._alias_section_node(node)
            return f"{self._nindent}{node.lhs.name} = {node.rhs.name};\n"
        return super().assignment_node(node)

    def _alias_section_node(self, node: Assignment) -> str:
        """Aim a handle at a contiguous rank-1 section: a ``subview``.

        A pointer that the callee aimed at a whole dummy is aimed, once the
        callee is inlined against a section actual, at ``field(w3_idx:w3_idx
        + nlayers - 1)``. Written as a handle copy of ``field`` that would
        read the first column for every column and compile without a word
        -- which is what the vertical FFSL regions did until 2026-09-13.
        The section is a ``Kokkos::subview`` instead, whose element ``0`` is
        the section's first, so the reads through the handle keep the
        origin the alias copied from its target: one.

        Only a rank-1 section with unit stride can be said this way; a
        strided section is not a ``subview`` of a ``LayoutLeft`` View and a
        higher rank one has no one origin to keep. Either is refused here
        as the last line, the transformation having refused it first.

        :param node: the pointer assignment whose right-hand side is a
            section.

        :returns: the subview assignment, indented for the body.

        :raises VisitorError: if the section is not rank-1 with unit stride,
            or its array has no View description.
        """
        target = node.rhs
        indices = target.indices
        if len(indices) != 1 or not isinstance(indices[0], Range):
            raise VisitorError(
                f"KokkosWriter cannot aim the handle '{node.lhs.name}' at "
                f"'{target.debug_string()}': only a rank-1 section of the "
                f"target can be a subview.")
        section = indices[0]
        if not (isinstance(section.step, Literal)
                and section.step.value == "1"):
            raise VisitorError(
                f"KokkosWriter cannot aim the handle '{node.lhs.name}' at "
                f"'{target.debug_string()}': a strided section cannot be a "
                f"subview.")
        try:
            view = self._views[target.name]
        except KeyError as err:
            raise VisitorError(
                f"Array '{target.name}' has no Kokkos View "
                f"description.") from err
        start = self._visit(section.start)
        stop = self._visit(section.stop)
        offset = view.index_offsets[0] if view.index_offsets else ""
        if offset:
            start = f"({start} - {offset})"
            stop = f"({stop} - {offset})"
        return (f"{self._nindent}{node.lhs.name} = Kokkos::subview("
                f"{target.name}, Kokkos::pair<int, int>({start}, {stop} + 1));"
                f"\n")

    def reference_node(self, node: Reference) -> str:
        """Emit a name, subscripting it where the region made it per-cell.

        Almost every reference is its own name and nothing else, which is
        what :py:class:`~psyclone.psyir.backend.c.CWriter` writes. The
        exception is a formal the kernel declares as a *scalar* and the
        region describes as a View: LFRic gives a stencil's size that way,
        one value per cell, and the region takes the whole array and
        subscripts it by the cell the thread is on. The kernel's own text
        names it bare, here and in every expression it reaches, so this is
        the one place that difference can be applied -- and applying it in
        only some of those places would read another cell's stencil rather
        than fail to compile.

        Such a View is recognised by carrying region indices and no kernel
        indices at all, which is exactly what a scalar formal made per-cell
        has: no dimension the kernel subscripts and one the region does.
        Every other View has at least one kernel index and is reached through
        :py:meth:`~psyclone.psyir.backend.kokkos_array_expression_mixin.\
KokkosArrayExpressionMixin.arrayreference_node` instead.

        :param node: the reference in the captured body.

        :returns: the name, subscripted by the region's own indices where the
            region described the scalar as a per-cell View.
        """
        indices = self._per_cell_indices(node)
        if indices:
            return f"{node.name}({', '.join(indices)})"
        return super().reference_node(node)

    def _per_cell_indices(self, node: Reference):
        """The region's own indices, where a scalar formal was made per-cell.

        Such a View carries region indices and no kernel indices at all,
        which is exactly what a scalar formal made per-cell has: no dimension
        the kernel subscripts and one the region does. Every other View has
        at least one kernel index and is reached through
        :py:meth:`~psyclone.psyir.backend.kokkos_array_expression_mixin.\
KokkosArrayExpressionMixin.arrayreference_node` instead.

        :param node: the reference being written.

        :returns: the indices to subscript the name by, or ``None`` where the
            reference is not a scalar the region made per-cell.
        :rtype: Optional[Tuple[str, ...]]
        """
        view = self._views.get(node.name)
        if (isinstance(view, KokkosView) and view.extra_indices
                and not view.index_offsets and not node.children):
            return view.extra_indices
        return None

    def _is_fixed_at_entry(self, expr, node) -> bool:
        """Judge a loop bound knowing which scalars are really View reads.

        :py:class:`~psyclone.psyir.backend.c.CWriter` leaves a reference to a
        scalar the body does not assign in the loop header, because nothing
        in the tree can change it. That is true of the kernel's own text and
        not of this region's: a scalar formal the region describes as a
        per-cell View is written as ``stencil_size(cell)``, a global read of
        staged memory, and re-reading that on every trip is the very thing
        this writer must not do. LFRic gives a stencil's size that way, so
        three of the prototype's captured regions turn on this.

        :param expr: the bound expression being judged.
        :type expr: :py:class:`psyclone.psyir.nodes.Node`
        :param node: the loop the expression is a bound of.
        :type node: :py:class:`psyclone.psyir.nodes.Loop`

        :returns: whether the expression may be left in the loop header.
        :rtype: bool
        """
        if isinstance(expr, Reference) and self._per_cell_indices(expr):
            return False
        return super()._is_fixed_at_entry(expr, node)

    @staticmethod
    def _argument_declaration(argument):
        """Return one declaration in the generated C ABI.

        :param argument: the scalar or View to declare.
        :type argument: Union[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScalar`,
            :py:class:`psyclone.psyir.backend.kokkos.KokkosView`]

        :returns: the C declaration, a value for a scalar and a pointer to
            caller-owned storage for a View.
        :rtype: str
        """
        if isinstance(argument, KokkosScalar):
            return f"const {argument.c_type} {argument.name}"
        const = "const " if argument.read_only else ""
        return f"{const}{argument.c_type} *{argument.data_name}"

    @staticmethod
    def _alias_declaration(alias, views):
        """Return the declaration of one alias handle.

        A handle in ``Kokkos::AnonymousSpace`` rather than the ``decltype``
        of its first target, so that a pointer aimed at a kernel argument in
        one branch and at a kernel-local array in the other is one handle
        rather than a refusal. ``AnonymousSpace`` is a memory space declared
        assignable from and to every other -- that is all
        ``Kokkos_AnonymousSpace.hpp`` says about it -- so a ``View`` in it
        holds a ``View`` of the region's space and a ``View`` of the launch's
        scratch alike, provided they agree in element type, rank and layout.
        Every ``View`` this back-end writes is ``LayoutLeft``, and the
        targets' agreement on the rest is what
        :py:meth:`_validate_alias` checks.

        **The element type is ``const`` where ANY target is ``const``**, and
        not where the first one happens to be. A ``View`` of ``T`` cannot be
        assigned from a ``View`` of ``const T`` -- that is the one direction
        Kokkos refuses, and it refuses it at the assignment rather than at
        the declaration -- while the reverse is allowed and costs nothing.
        So a pointer aimed at a kernel-local column and at a read-only
        argument is a handle of ``const T`` whichever branch the body writes
        first, and both of its assignments compile. Reading through such a
        handle is all the body may do, which
        :py:meth:`_validate_alias` is where it is required.

        The rank and the memory traits are still the first target's, which
        is what ``decltype`` took them from. The traits need no widening
        with the constness: ``ReadOnly`` -- ``Unmanaged | RandomAccess`` --
        is carried only by a read-only argument, whose element type is
        already ``const``, so a first target that gives ``Unmanaged`` gives
        it whether or not a later target made the handle ``const``.

        :param alias: the alias to declare.
        :type alias: :py:class:`psyclone.psyir.backend.kokkos.KokkosAlias`
        :param views: the region's arrays keyed by name, in which the alias's
            targets are described.
        :type views: Dict[str, Union[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosView`,
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScratch`]]

        :returns: the declaration, without indentation.
        :rtype: str
        """
        target = views[alias.targets[0]]
        read_only = any(getattr(views[name], "read_only", False)
                        for name in alias.targets)
        random_access = (isinstance(target, KokkosView)
                         and target.random_access)
        const = "const " if read_only else ""
        traits = "ReadOnly" if random_access else "Unmanaged"
        rank = "*" * len(target.extents)
        return (
            f"Kokkos::View<{const}{target.c_type}{rank}, Kokkos::LayoutLeft, "
            f"Kokkos::AnonymousSpace, {traits}> {alias.name};")


__all__ = ["KokkosAlias", "KokkosColourMap", "KokkosConstant",
           "KokkosRegion",
           "KokkosScalar",
           "KokkosScratch", "KokkosView",
           "KokkosWriter", "extent_names", "is_extent", "is_offset"]
