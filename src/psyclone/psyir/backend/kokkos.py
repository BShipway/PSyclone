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

from psyclone.psyir.backend.c import CWriter
from psyclone.psyir.backend.kokkos_array_expression import KokkosScratch
from psyclone.psyir.backend.kokkos_array_expression_mixin import (
    KokkosArrayExpressionMixin)
from psyclone.psyir.backend.kokkos_intrinsics_mixin import (
    KokkosIntrinsicsMixin)
from psyclone.psyir.backend.kokkos_constant import KokkosConstant
from psyclone.psyir.backend.kokkos_region import (
    KokkosColourMap, KokkosRegion, KokkosScalar, KokkosView,
    extent_names, is_extent, is_identifier, is_offset)
from psyclone.psyir.backend.kokkos_team_scalars import (
    team_private_scalars)
from psyclone.psyir.backend.kokkos_launch import (
    hierarchical_launch, range_launch, team_launch)
from psyclone.psyir.nodes import (
    CodeBlock, KernelSchedule, Literal, Loop, Reference)


class KokkosWriter(KokkosIntrinsicsMixin, KokkosArrayExpressionMixin,
                   CWriter):
    """Generate a C++/Kokkos translation unit for a captured region.

    The intrinsics and the handling of array-valued expressions are inherited
    rather than written here, from
    :py:mod:`~psyclone.psyir.backend.kokkos_intrinsics_mixin` and
    :py:mod:`~psyclone.psyir.backend.kokkos_array_expression_mixin`, the
    lowering the second of them drives being
    :py:mod:`~psyclone.psyir.backend.kokkos_array_expression`. Both mixins
    precede :py:class:`~psyclone.psyir.backend.c.CWriter` in the bases so that
    their handlers are found first and fall through to the C writer's by
    ``super()``.
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

        One of three launch shapes is generated. A region naming any
        :py:attr:`KokkosRegion.parallel_loops` takes the hierarchical launch;
        otherwise one describing any :py:attr:`KokkosRegion.scratch` takes the
        flat team launch; otherwise the range launch. Neither of the older two
        reads the fields it does not select on, so a region built before they
        existed is generated exactly as it was then; see
        :py:func:`~psyclone.psyir.backend.kokkos_launch.range_launch`,
        :py:func:`~psyclone.psyir.backend.kokkos_launch.team_launch` and
        :py:func:`~psyclone.psyir.backend.kokkos_launch.hierarchical_launch`.

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
        self._kind_types = dict(region.kind_types)
        self._parallel_loops = region.parallel_loops

        signature = ",\n    ".join(
            self._argument_declaration(argument)
            for argument in region.arguments)
        views = "\n".join(
            self._view_declaration(argument)
            for argument in region.arguments
            if isinstance(argument, KokkosView))
        # Both team shapes need the policy and its member type. Only a region
        # with scratch needs the space its Views are placed in, and the
        # hierarchical shape is reached without scratch, so that alias is
        # conditioned separately rather than riding along unused.
        team_aliases = "".join(
            f"  {alias}\n" for alias in (
                "using TeamPolicy = Kokkos::TeamPolicy<>;",
                "using TeamMember = TeamPolicy::member_type;",
            )) if region.scratch or region.parallel_loops else ""
        if region.scratch:
            team_aliases += (
                "  using ScratchSpace = "
                "Kokkos::DefaultExecutionSpace::scratch_memory_space;\n")

        # The flat team shape nests the body one level deeper, inside the
        # TeamThreadRange lambda. The hierarchical one does not: its body sits
        # directly in the functor, as the range shape's does.
        scratch_names = {item.name for item in region.scratch}
        self._depth = 3 if region.scratch and not region.parallel_loops else 2
        local_declarations = "".join(
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
            if symbol.name not in scratch_names
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
            colours = region.colour_map
            local_declarations = (
                f"{self._nindent}const int {region.cell_index} = "
                f"{colours.name}({colours.colour} - 1, {colours.index}) "
                "- 1;\n") + local_declarations
        body = "".join(
            self._visit(child) for child in region.schedule.children)
        constant_indent, self._depth = self._nindent, 0
        self._views, self._kind_types = {}, {}
        self._parallel_loops = ()
        self._private_scalars = {}

        # Inside the body, not at file scope: nvcc will not read a namespace
        # scope array from device code. First, since a constant reads nothing.
        local_declarations = "".join(
            f"{constant_indent}const {item.c_type} {item.name}"
            f"[{len(item.values)}] = "
            f"{{{', '.join(self._visit(value) for value in item.values)}}};\n"
            for item in region.constants) + local_declarations

        if region.parallel_loops:
            launch = hierarchical_launch(region, local_declarations, body)
        elif region.scratch:
            launch = team_launch(region, local_declarations, body)
        else:
            launch = range_launch(region, local_declarations, body)

        return (
            "#include <Kokkos_Core.hpp>\n\n"
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
            "}\n")

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
        view = self._views.get(node.name)
        if (isinstance(view, KokkosView) and view.extra_indices
                and not view.index_offsets and not node.children):
            return f"{node.name}({', '.join(view.extra_indices)})"
        return super().reference_node(node)

    def _validate(self, region):
        """Reject incomplete or unsupported region descriptions.

        :param region: the region description to check.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

        :raises TypeError: if ``region`` is not a
            :py:class:`KokkosRegion`, if its schedule is not a
            :py:class:`~psyclone.psyir.nodes.KernelSchedule`, if an argument
            is neither a :py:class:`KokkosScalar` nor a
            :py:class:`KokkosView`, if an argument's or a kind's C type is not
            in :py:attr:`_SUPPORTED_TYPES`, if a View's index offsets are
            not integers, if a :py:attr:`KokkosRegion.parallel_loops` entry is
            not a :py:class:`~psyclone.psyir.nodes.Loop`, if a constant is
            not a :py:class:`KokkosConstant` of a supported C type, or if the
            region's
            :py:attr:`KokkosRegion.team_size` is neither ``None`` nor an
            ``int`` -- ``bool`` among them, since ``TeamPolicy(ncells, True)``
            is a legal team of one that nothing downstream would report.
        :raises ValueError: if the region's name, its cell count, an argument
            name, a kind name or a View's data name or region indices are not
            C++ identifiers; if a View's or a scratch array's extent is not an
            integer expression over named sizes; if the schedule contains a
            :py:class:`~psyclone.psyir.nodes.CodeBlock`; if two arguments
            share a C ABI name; if the cell count is not itself a scalar
            argument; if the cell position breaks the contract
            :py:meth:`_validate_cell_position` states; if a colour map
            breaks the contract :py:meth:`_validate_colour_map` states; if a
            kernel argument
            has no description; if a View
            breaks the ownership or dimensional contract
            :py:meth:`_validate_view` states; if a scratch array breaks the
            contract :py:meth:`_validate_scratch` states; if a parallel loop
            is not in the region's schedule, is nested inside another of them,
            or has a step other than the literal ``1``; if a constant has no
            values or is not one dimensional; or if the team size is not
            positive.
        """
        # A validator is a list of checks, and reads better as one than as an
        # arbitrary split into halves that share every name they compute.
        # pylint: disable=too-many-branches, too-many-statements
        if not isinstance(region, KokkosRegion):
            raise TypeError(
                "KokkosWriter expects a KokkosRegion but found "
                f"'{type(region).__name__}'.")
        if not isinstance(region.schedule, KernelSchedule):
            raise TypeError("KokkosRegion schedule must be a KernelSchedule.")
        if not is_identifier(region.name):
            raise ValueError(
                f"Kokkos region name '{region.name}' is not a C++ identifier.")
        if not is_identifier(region.cell_count):
            raise ValueError(
                f"Cell count '{region.cell_count}' is not a C++ identifier.")
        if not is_identifier(region.cell_index):
            raise ValueError(
                f"Cell index '{region.cell_index}' is not a C++ identifier.")
        if region.schedule.walk(CodeBlock):
            raise ValueError("Kokkos regions cannot contain a CodeBlock.")

        for kind, c_type in region.kind_types:
            if not is_identifier(kind):
                raise ValueError(
                    f"Kokkos kind name '{kind}' is not a C++ identifier.")
            if c_type not in self._SUPPORTED_TYPES:
                raise TypeError(
                    f"Kokkos kind '{kind}' has unsupported C type "
                    f"'{c_type}'.")

        abi_names = set()
        view_names = set()
        scalar_names = set()
        for argument in region.arguments:
            if not isinstance(argument, (KokkosScalar, KokkosView)):
                raise TypeError(
                    "KokkosRegion arguments must be KokkosScalar or "
                    "KokkosView instances, found "
                    f"'{type(argument).__name__}'.")
            if argument.c_type not in self._SUPPORTED_TYPES:
                raise TypeError(
                    f"Kokkos argument '{argument.name}' has unsupported C "
                    f"type '{argument.c_type}'.")
            if not is_identifier(argument.name):
                raise ValueError(
                    f"Kokkos argument name '{argument.name}' is invalid.")
            if isinstance(argument, KokkosScalar):
                abi_name = argument.name
                scalar_names.add(argument.name)
            else:
                abi_name = argument.data_name
                view_names.add(argument.name)
                self._validate_view(argument)
            if abi_name in abi_names:
                raise ValueError(f"Duplicate C ABI argument '{abi_name}'.")
            abi_names.add(abi_name)

        if region.cell_count not in scalar_names:
            raise ValueError(
                f"Cell count '{region.cell_count}' is not a scalar argument.")
        if region.colour_map is not None:
            self._validate_colour_map(region, view_names, scalar_names)

        schedule_arguments = {
            symbol.name
            for symbol in region.schedule.symbol_table.argument_list
        }
        provided = scalar_names | view_names
        if region.cell_position is not None:
            self._validate_cell_position(
                region, schedule_arguments, provided)
            # Described by declaration rather than by argument, so it counts
            # as provided for the completeness check below.
            provided = provided | {region.cell_position}
        missing = schedule_arguments - provided
        if missing:
            raise ValueError(
                "Kokkos region does not describe kernel arguments: "
                f"{', '.join(sorted(missing))}.")

        # Scratch is deliberately not folded into the loop above: it is not a
        # kernel formal, so it takes no part in the C ABI or in the check
        # that every formal was described.
        used_names = abi_names | view_names
        for item in region.scratch:
            self._validate_scratch(item, scalar_names, used_names)
            used_names.add(item.name)
        # A carried constant is one dimensional, because that is what a
        # Fortran parameter array the region can index in C storage order is.
        for item in region.constants:
            if (not isinstance(item, KokkosConstant)
                    or item.c_type not in self._SUPPORTED_TYPES):
                raise TypeError(
                    "KokkosRegion constants must be KokkosConstant instances "
                    f"of a supported C type, found '{item}'.")
            if not item.values or len(item.index_offsets) != 1:
                raise ValueError(
                    f"Kokkos constant '{item.name}' must have at least one "
                    "value and exactly one index offset.")

        self._validate_launch(region)

    def _validate_cell_position(self, region, formals, described):
        """Reject a cell position the region could not correctly declare.

        Checked rather than trusted for the reason
        :py:meth:`_validate_launch` gives: none of the four mistakes stops a
        build. A name that is not a formal declares a local nothing reads; a
        name the region also passes puts a parameter and a local of the same
        name in one scope, which C++ resolves in favour of the local; and a
        name equal to the launch index generates ``const int cell = cell + 1``,
        which initialises an object from itself.

        :param region: the region description to check.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
        :param formals: the names of the schedule's kernel arguments.
        :type formals: Set[str]
        :param described: the names ``region.arguments`` already covers.
        :type described: Set[str]

        :raises ValueError: if the cell position is not a C++ identifier, is
            not one of the schedule's kernel arguments, is also described as a
            region argument, or is the launch's own cell index.
        """
        position = region.cell_position
        if not is_identifier(position):
            raise ValueError(
                f"Cell position '{position}' is not a C++ identifier.")
        if position == region.cell_index:
            raise ValueError(
                f"Cell position '{position}' is also the launch's cell "
                "index.")
        if position not in formals:
            raise ValueError(
                f"Cell position '{position}' is not a kernel argument.")
        if position in described:
            raise ValueError(
                f"Cell position '{position}' is also described as a region "
                "argument.")

    @staticmethod
    def _validate_launch(region):
        """Validate the loops and the team size the hierarchical launch takes.

        These are checked rather than trusted because none of the four
        mistakes below announces itself downstream: a loop from another
        schedule is never matched by identity and silently generates a serial
        ``for``, and the other three generate C++ that compiles and means
        something the Fortran did not.

        :param region: the region description to check.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

        :raises TypeError: if a ``parallel_loops`` entry is not a
            :py:class:`~psyclone.psyir.nodes.Loop`, or if ``team_size`` is
            neither ``None`` nor an ``int``.
        :raises ValueError: if a parallel loop is not in the region's
            schedule, is nested inside another of them, or has a step other
            than the literal ``1``; or if the team size is not positive.
        """
        for entry in region.parallel_loops:
            if not isinstance(entry, Loop):
                raise TypeError(
                    "KokkosRegion parallel_loops must be Loop instances, "
                    f"found '{type(entry).__name__}'.")
            if not any(entry is loop
                       for loop in region.schedule.walk(Loop)):
                raise ValueError(
                    f"Kokkos parallel loop over '{entry.variable.name}' is "
                    "not in the region's schedule.")
            ancestor = entry.ancestor(Loop)
            while ancestor is not None:
                if any(ancestor is other
                       for other in region.parallel_loops):
                    raise ValueError(
                        f"Kokkos parallel loop over '{entry.variable.name}' "
                        "is nested inside another parallel loop.")
                ancestor = ancestor.ancestor(Loop)
            step = entry.step_expr
            if not isinstance(step, Literal) or step.value != "1":
                raise ValueError(
                    f"Kokkos parallel loop over '{entry.variable.name}' has "
                    "a step that is not the literal 1; TeamVectorRange has "
                    "no stride.")

        if region.team_size is None:
            return
        if isinstance(region.team_size, bool) or not isinstance(
                region.team_size, int):
            raise TypeError(
                "KokkosRegion team_size must be None or an int, found "
                f"'{type(region.team_size).__name__}'.")
        if region.team_size <= 0:
            raise ValueError(
                f"KokkosRegion team_size must be positive, found "
                f"{region.team_size}.")

    def _validate_scratch(self, scratch, scalar_names, used_names):
        """Validate one kernel-local array placed in team scratch.

        :param scratch: the scratch description to check.
        :type scratch:
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScratch`
        :param scalar_names: the names of the region's scalar arguments, which
            are the only extents the generated C++ can name.
        :type scalar_names: Set[str]
        :param used_names: every name already taken by an argument, a View or
            an earlier scratch array.
        :type used_names: Set[str]

        :raises ValueError: if the scratch is not a
            :py:class:`KokkosScratch`; if its name is not a C++ identifier or
            is already taken; if it has no extents, an extent that is not an
            integer expression over named sizes, or an extent naming something
            that is not a scalar argument of the region; or if its rank does
            not match the index offsets supplied for it.
        :raises TypeError: if its C type is not in
            :py:attr:`_SUPPORTED_TYPES`, or an index offset is neither an
            integer nor an integer expression over named sizes.
        """
        if not isinstance(scratch, KokkosScratch):
            raise ValueError(
                "KokkosRegion scratch must be KokkosScratch instances, found "
                f"'{type(scratch).__name__}'.")
        if not is_identifier(scratch.name):
            raise ValueError(
                f"Kokkos scratch name '{scratch.name}' is invalid.")
        if scratch.name in used_names:
            raise ValueError(
                f"Kokkos scratch '{scratch.name}' collides with an existing "
                "region name.")
        if scratch.c_type not in self._SUPPORTED_TYPES:
            raise TypeError(
                f"Kokkos scratch '{scratch.name}' has unsupported C type "
                f"'{scratch.c_type}'.")
        if not scratch.extents:
            raise ValueError(
                f"Kokkos scratch '{scratch.name}' must have extents.")
        for extent in scratch.extents:
            if not is_extent(extent):
                raise ValueError(
                    f"Kokkos scratch '{scratch.name}' has extent '{extent}' "
                    "which is not an integer expression over named sizes.")
            # Sorted for the same reason the transformation sorts: the
            # refusal names one offender, and a set has no fixed order.
            for name in sorted(extent_names(extent)):
                if name not in scalar_names:
                    raise ValueError(
                        f"Kokkos scratch '{scratch.name}' has extent "
                        f"'{extent}', which is sized from '{name}' rather "
                        "than from a scalar argument.")
        if len(scratch.extents) != len(scratch.index_offsets):
            raise ValueError(
                f"Kokkos scratch '{scratch.name}' dimensions do not match its "
                "kernel indices.")
        if not all(is_offset(offset) for offset in scratch.index_offsets):
            raise TypeError(
                f"Kokkos scratch '{scratch.name}' index offsets must be "
                "integers or integer expressions over named sizes.")

    def _validate_colour_map(self, region, view_names, scalar_names):
        """Validate the names a coloured launch generates and reads.

        Checked rather than trusted for the reason
        :py:meth:`_validate_cell_position` gives, and with one addition that
        is worse than any of those: a colour map that is not passed, or is
        passed as something other than a View, generates a lookup of a name
        the translation unit does not hold, and the region would run every
        cell of every colour at once if the lookup were quietly dropped.

        :param region: the region whose colour map is to be checked.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
        :param view_names: the names ``region.arguments`` passes as Views.
        :type view_names: Set[str]
        :param scalar_names: the names ``region.arguments`` passes as
            scalars.
        :type scalar_names: Set[str]

        :raises ValueError: if any of the three names is not a C++
            identifier; if the map is not passed as a View or the colour is
            not passed as a scalar; if the launch's own index is the cell
            index, which would declare the cell from itself; or if that index
            is also a region argument, which the declaration would shadow.
        """
        colours = region.colour_map
        for description, name in (("map", colours.name),
                                  ("colour", colours.colour),
                                  ("index", colours.index)):
            if not is_identifier(name):
                raise ValueError(
                    f"Kokkos colour {description} '{name}' is not a C++ "
                    "identifier.")
        if colours.name not in view_names:
            raise ValueError(
                f"Kokkos colour map '{colours.name}' is not a View argument.")
        if colours.colour not in scalar_names:
            raise ValueError(
                f"Kokkos colour '{colours.colour}' is not a scalar argument.")
        if colours.index == region.cell_index:
            raise ValueError(
                f"Kokkos colour index '{colours.index}' is also the region's "
                "cell index, so the cell would be declared from itself.")
        if colours.index in view_names | scalar_names:
            raise ValueError(
                f"Kokkos colour index '{colours.index}' is also a region "
                "argument.")

    def _validate_view(self, view):
        """Validate the ownership and dimensional contract for one View.

        :param view: the View description to check.
        :type view: :py:class:`psyclone.psyir.backend.kokkos.KokkosView`

        :raises ValueError: if the View is managed, so would own LFRic
            storage; if its data name or region indices are not C++
            identifiers; if an extent is not an integer expression over named
            sizes; if its rank does not match the kernel and region indices
            supplied for it; if it is writable while asking for
            ``RandomAccess``; or if it is read only while asking for atomic
            updates, which would be a description of an update that cannot
            happen.
        :raises TypeError: if an index offset is neither an integer nor an
            integer expression over named sizes.
        """
        if view.managed:
            raise ValueError(f"Kokkos View '{view.name}' must be unmanaged.")
        if not is_identifier(view.data_name):
            raise ValueError(
                f"Kokkos View data name '{view.data_name}' is invalid.")
        if not view.extents or not all(
                is_extent(extent) for extent in view.extents):
            raise ValueError(
                f"Kokkos View '{view.name}' must have extents that are "
                "integer expressions over named sizes.")
        if len(view.extents) != (
                len(view.index_offsets) + len(view.extra_indices)):
            raise ValueError(
                f"Kokkos View '{view.name}' dimensions do not match its "
                "kernel and region indices.")
        if not all(is_offset(offset) for offset in view.index_offsets):
            raise TypeError(
                f"Kokkos View '{view.name}' index offsets must be integers or "
                "integer expressions over named sizes.")
        if not all(is_identifier(index)
                   for index in view.extra_indices):
            raise ValueError(
                f"Kokkos View '{view.name}' has an invalid region index.")
        if view.random_access and not view.read_only:
            raise ValueError(
                f"Kokkos View '{view.name}' uses RandomAccess but is "
                "writable.")
        if view.atomic and view.read_only:
            raise ValueError(
                f"Kokkos View '{view.name}' is atomic but read only.")

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
    def _view_declaration(view):
        """Return an unmanaged View declaration.

        :param view: the View to declare over storage the caller owns.
        :type view: :py:class:`psyclone.psyir.backend.kokkos.KokkosView`

        :returns: the declaration, indented for the region body.
        :rtype: str
        """
        const = "const " if view.read_only else ""
        rank = "*" * len(view.extents)
        traits = "ReadOnly" if view.random_access else "Unmanaged"
        extents = ", ".join(view.extents)
        return (
            f"  Kokkos::View<{const}{view.c_type}{rank}, "
            f"Kokkos::LayoutLeft, MemorySpace, {traits}> {view.name}("
            f"{view.data_name}, {extents});")


__all__ = ["KokkosColourMap", "KokkosConstant", "KokkosRegion",
           "KokkosScalar",
           "KokkosScratch", "KokkosView",
           "KokkosWriter", "extent_names", "is_extent", "is_offset"]
