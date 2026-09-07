# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""A deliberately small Kokkos backend for whole LFRic loop regions."""

from dataclasses import dataclass
import re
from typing import Optional, Tuple, Union

from psyclone.psyir.backend.c import CWriter
from psyclone.psyir.backend.kokkos_intrinsics_mixin import (
    KokkosIntrinsicsMixin)
from psyclone.psyir.backend.kokkos_constant import KokkosConstant
from psyclone.psyir.backend.kokkos_launch import (
    hierarchical_launch, range_launch, team_launch)
from psyclone.psyir.nodes import (
    ArrayReference, CodeBlock, KernelSchedule, Literal, Loop, Reference)
from psyclone.psyir.symbols import ArrayType


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extent_names(value):
    """Return the identifiers an extent expression is sized from.

    An integer literal contributes nothing, so ``"(nlayers + 1)"`` gives
    ``{"nlayers"}`` and ``"4"`` gives the empty set. Callers use this to ask
    whether an extent can be evaluated where it is written, without having to
    parse the expression themselves.

    :param value: the candidate extent, which need not be a string.

    :returns: every C++ identifier appearing in it.
    :rtype: set[str]
    """
    if not isinstance(value, str):
        return set()
    return set(_IDENTIFIER.findall(value))


def is_offset(value):
    """Return whether ``value`` may be written as an index offset.

    An offset is the declared origin of one dimension, subtracted from every
    Fortran subscript of that dimension. It is an integer where the origin is
    a constant -- ``1`` for the Fortran default -- and an integer expression
    over named sizes where it is not, as for an array declared
    ``dimension(-stencil:stencil)``. The expression grammar is
    :py:func:`is_extent`'s, and deliberately the same one: an origin and an
    extent are two readings of one declaration, and a rule that admitted a
    shape into the extent and refused it in the origin would size a View
    correctly and index it from the wrong place.

    :param value: the candidate offset, which need not be a string.

    :returns: whether it can be written into generated C++ as an offset.
    :rtype: bool
    """
    return isinstance(value, int) or is_extent(value)


def is_extent(value):
    """Return whether ``value`` may be written as a Kokkos extent.

    An extent is an integer expression over named sizes, so a bare name is
    accepted as before and so are ``max_length``, ``4`` and
    ``(nlayers + 1)``. Division is refused rather than merely unsupported:
    Fortran and C++ can disagree about the rounding of an integer division,
    and an extent is one of the few places where that disagreement would
    produce a wrongly sized allocation instead of a compile error.

    :param value: the candidate extent, which need not be a string.

    :returns: whether it can be written into generated C++ as an extent.
    :rtype: bool
    """
    if not isinstance(value, str) or not value.strip():
        return False
    if not re.fullmatch(r"[A-Za-z0-9_ ()+\-*]+", value):
        return False
    depth = 0
    for character in value:
        depth += (character == "(") - (character == ")")
        if depth < 0:
            return False
    if depth:
        return False
    # Split on the operators rather than searching for names, so that a
    # malformed token such as ``4nlayers`` is seen whole and refused instead
    # of reading as a literal beside an identifier.
    for token in re.split(r"[ ()+\-*]+", value):
        if token and not (token.isdigit()
                          or _IDENTIFIER.fullmatch(token)):
            return False
    return True


@dataclass(frozen=True)
class KokkosScalar:
    """A scalar on the generated C ABI."""

    name: str
    c_type: str


@dataclass(frozen=True)
class KokkosView:
    """An unmanaged Kokkos View over storage owned by the caller."""
    # pylint: disable=too-many-instance-attributes

    name: str
    data_name: str
    c_type: str
    #: One integer expression per dimension, over the region's scalar
    #: arguments and integer literals: ``nlayers``, ``4``, ``(nlayers + 1)``.
    #: They are emitted into the generated C++ verbatim.
    extents: Tuple[str, ...]
    #: The declared origin of each dimension: the value subtracted from a
    #: Fortran subscript to reach the zero-based View element it names. An
    #: integer, or an integer expression over the region's scalar arguments
    #: for an array whose origin is not a constant. ``1`` for the Fortran
    #: default, ``0`` for an array declared ``dimension(0:nlayers)``.
    index_offsets: Tuple[Union[int, str], ...] = ()
    extra_indices: Tuple[str, ...] = ()
    read_only: bool = False
    random_access: bool = False
    managed: bool = False


@dataclass(frozen=True)
class KokkosScratch:
    """A kernel-local array placed in Kokkos team scratch.

    A Fortran automatic local such as ``real(r_def), dimension(nlayers) ::
    x_new`` crosses no interface, so it is described here rather than among
    the region's arguments: it must not appear in the generated C ABI, and it
    is not a kernel formal the region has to account for. Its extents are
    integer expressions over the region's scalar arguments, such as
    ``nlayers``, ``4`` or ``(nlayers + 1)``, which is what lets the generated
    C++ size it: a scratch size is computed on the host before the launch,
    where only those scalars are in scope.

    ``index_offsets`` and ``extra_indices`` are carried, and the latter is
    always empty, so that one array-reference table can hold both Views and
    scratch and be read without a type test.
    """

    name: str
    c_type: str
    extents: Tuple[str, ...]
    #: As :py:attr:`KokkosView.index_offsets`.
    index_offsets: Tuple[Union[int, str], ...] = ()
    extra_indices: Tuple[str, ...] = ()


@dataclass(frozen=True)
class KokkosRegion:
    """All information required to generate one Kokkos translation unit."""
    # A description carries as many fields as the thing it describes has
    # parts, and splitting them into sub-objects would only move the count.
    # pylint: disable=too-many-instance-attributes

    name: str
    schedule: KernelSchedule
    cell_count: str
    # Spelt out rather than given a module-level alias: ``autoapi`` renders
    # every module variable as a literal block and Sphinx then appends its
    # own "alias of" line unindented, which fails the ``-W`` doc build.
    arguments: Tuple[Union[KokkosScalar, KokkosView], ...]
    #: The name the launch gives its own cell index: the ``RangePolicy``
    #: lambda's parameter, or the value the team launch computes from the
    #: league rank. It defaults to ``cell``, so a region built before this
    #: field existed generates exactly the source it generated then. A caller
    #: renames it when the kernel already declares ``cell`` itself, because a
    #: lambda parameter and a body declaration share one C++ scope, so the
    #: collision is rejected by the compiler rather than silently miscompiled.
    #: The per-cell Views' ``extra_indices`` have to name it too; the writer
    #: does not rewrite them.
    cell_index: str = "cell"
    #: One ``(Fortran kind name, C type)`` pair per kind the region's body
    #: mentions, such as ``("r_solver", "float")``. The region's arguments
    #: carry their own C types, but its locals and its literals cross no
    #: interface and would otherwise be generated at the C writer's default
    #: width -- silently promoting a single-precision kernel to double.
    #: Empty means "generate as the C writer would", which is what every
    #: region built before this field existed did.
    kind_types: Tuple[Tuple[str, str], ...] = ()
    #: One :py:class:`KokkosScratch` per kernel-local automatic array. This
    #: field selects the launch shape: empty gives the ``RangePolicy`` region
    #: generated for every capture before scratch existed, byte for byte,
    #: while a non-empty tuple gives a ``TeamPolicy`` region carrying
    #: per-thread scratch. A kernel with no local arrays has no scratch to
    #: place, so it keeps the simpler launch.
    scratch: Tuple[KokkosScratch, ...] = ()
    #: The loops of the region's own schedule that are to be spread over the
    #: team. A non-empty tuple selects the hierarchical launch -- one team per
    #: cell, with each of these loops rendered as a ``TeamVectorRange``
    #: ``parallel_for`` -- and overrides the selection by :py:attr:`scratch`;
    #: an empty tuple leaves that selection in force, so a region built before
    #: this field existed generates exactly the source it generated then.
    #: Which loops may be spread is a dependence judgement made by the driving
    #: transformation, not here; the writer checks only that what it is given
    #: is a set of unnested unit-stride loops it can find in the schedule.
    parallel_loops: Tuple[Loop, ...] = ()
    #: The name of the kernel formal carrying LFRic's cell index, or ``None``
    #: for a kernel that has none. LFRic passes that index to every kernel
    #: taking an operator, which uses it arithmetically to find its own slice
    #: of the operator's local stencil. It is the one formal the region
    #: declares rather than takes: the launch already knows which cell it is
    #: on, so the value is generated from :py:attr:`cell_index` inside the
    #: functor. Taking it across the ABI instead would compile and run, and
    #: give every cell whatever the caller passed once.
    cell_position: Optional[str] = None
    #: The team size the hierarchical launch asks for. ``None`` renders
    #: ``Kokkos::AUTO`` and lets the backend choose; a positive integer
    #: renders itself, which is how a host build reaches the team-level
    #: concurrency that ``AUTO`` sizes to one member. The flat shapes ignore
    #: it: the range launch has no team, and the flat team launch takes the
    #: size the backend recommends for its own functor.
    team_size: Optional[int] = None
    #: One :py:class:`KokkosConstant` per ``parameter`` array the body reads.
    #: Declared among the body's locals and taking no place on the ABI, so a
    #: region built before this field existed generates what it did then.
    constants: Tuple[KokkosConstant, ...] = ()


class KokkosWriter(KokkosIntrinsicsMixin, CWriter):
    """Generate a C++/Kokkos translation unit for a captured region.

    The intrinsics are inherited rather than written here, and the mixin
    precedes :py:class:`~psyclone.psyir.backend.c.CWriter` in the bases so
    that its handlers are found first and fall through to the C writer's by
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

        :param region: the captured region to generate.

        :returns: a complete C++ translation unit.

        :raises TypeError: as :py:meth:`_validate` does.
        :raises ValueError: as :py:meth:`_validate` does, and if the body
            indexes an array for which the region described neither a View nor
            scratch.
        """
        self._validate(region)
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
            if symbol.name not in scratch_names)
        if region.cell_position is not None:
            # First, and prepended here rather than in each launch shape: all
            # three place these declarations immediately after establishing
            # their own cell index, which is the only thing this one reads.
            local_declarations = (
                f"{self._nindent}const int {region.cell_position} = "
                f"{region.cell_index} + 1;\n") + local_declarations
        body = "".join(
            self._visit(child) for child in region.schedule.children)
        constant_indent, self._depth = self._nindent, 0
        self._views, self._kind_types = {}, {}
        self._parallel_loops = ()

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

    @staticmethod
    def _is_identifier(value):
        """Return whether ``value`` is a C++ identifier.

        :param value: the candidate name, which need not be a string.

        :returns: whether it can be written into generated C++ as a name.
        :rtype: bool
        """
        return isinstance(value, str) and bool(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))

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
            :py:meth:`_validate_cell_position` states; if a kernel argument
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
        if not self._is_identifier(region.name):
            raise ValueError(
                f"Kokkos region name '{region.name}' is not a C++ identifier.")
        if not self._is_identifier(region.cell_count):
            raise ValueError(
                f"Cell count '{region.cell_count}' is not a C++ identifier.")
        if not self._is_identifier(region.cell_index):
            raise ValueError(
                f"Cell index '{region.cell_index}' is not a C++ identifier.")
        if region.schedule.walk(CodeBlock):
            raise ValueError("Kokkos regions cannot contain a CodeBlock.")

        for kind, c_type in region.kind_types:
            if not self._is_identifier(kind):
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
            if not self._is_identifier(argument.name):
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
        if not self._is_identifier(position):
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
        if not self._is_identifier(scratch.name):
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

    def _validate_view(self, view):
        """Validate the ownership and dimensional contract for one View.

        :param view: the View description to check.
        :type view: :py:class:`psyclone.psyir.backend.kokkos.KokkosView`

        :raises ValueError: if the View is managed, so would own LFRic
            storage; if its data name or region indices are not C++
            identifiers; if an extent is not an integer expression over named
            sizes; if its rank does not match the kernel and region indices
            supplied for it; or if it is writable while asking for
            ``RandomAccess``.
        :raises TypeError: if an index offset is neither an integer nor an
            integer expression over named sizes.
        """
        if view.managed:
            raise ValueError(f"Kokkos View '{view.name}' must be unmanaged.")
        if not self._is_identifier(view.data_name):
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
        if not all(self._is_identifier(index)
                   for index in view.extra_indices):
            raise ValueError(
                f"Kokkos View '{view.name}' has an invalid region index.")
        if view.random_access and not view.read_only:
            raise ValueError(
                f"Kokkos View '{view.name}' uses RandomAccess but is "
                "writable.")

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

    def _kind_c_type(self, datatype):
        """Return the C type this region generates for a datatype's kind.

        :param datatype: the datatype whose kind is to be resolved.
        :type datatype: :py:class:`psyclone.psyir.symbols.DataType`

        :returns: the C type, or ``None`` if the datatype names no kind that
            the region described.
        :rtype: Optional[str]
        """
        if isinstance(datatype, ArrayType):
            datatype = datatype.elemental_type
        precision = getattr(datatype, "precision", None)
        if not isinstance(precision, Reference):
            return None
        return self._kind_types.get(precision.symbol.name)

    def gen_declaration(self, symbol) -> str:
        """Declare a symbol at the width its Fortran kind actually has.

        The C writer maps every real onto ``double``, which would promote a
        single-precision kernel's locals without saying so. A symbol whose
        kind the region did not describe is left to it: the counter
        :py:class:`~psyclone.psyir.transformations.ArrayAssignment2LoopsTrans`
        introduces for a lowered array section has no named kind, and must
        still be generated as ``int``.

        :param symbol: the symbol to declare.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the C declaration, without indentation or punctuation.
        :rtype: str
        """
        c_type = self._kind_c_type(symbol.datatype)
        if c_type is None:
            return super().gen_declaration(symbol)
        pointer = "* restrict " if symbol.is_array else ""
        return f"{c_type} {pointer}{symbol.name}"

    def loop_node(self, node) -> str:
        """Spread one of the region's chosen loops over the team.

        A loop the region named in :py:attr:`KokkosRegion.parallel_loops`
        becomes a ``TeamVectorRange`` ``parallel_for``; every other loop,
        including one nested inside this body, is left to
        :py:class:`~psyclone.psyir.backend.c.CWriter` as a serial ``for`` that
        each member runs on its own. The Fortran bound is inclusive and the
        Kokkos range is half-open, which is the ``+ 1`` on the stop
        expression.

        A ``team_barrier`` follows unconditionally. A statement after the loop
        may read what the loop wrote, and working out whether one does is a
        second dependence analysis this writer does not perform; under
        ``Kokkos::AUTO`` on the OpenMP backend the team has one member and the
        barrier costs nothing measurable.

        The lambda's parameter shadows the region-scope declaration of the
        loop variable, which stays because the same variable may also drive a
        serial loop in the same body. Shadowing a local with a lambda
        parameter is legal C++, and the generated code is not compiled with
        ``-Wshadow``.

        :param node: the loop in the captured body.
        :type node: :py:class:`psyclone.psyir.nodes.Loop`

        :returns: the ``TeamVectorRange`` launch and its barrier, or the
            serial ``for`` the C writer would have produced.
        :rtype: str
        """
        if not any(loop is node for loop in self._parallel_loops):
            return super().loop_node(node)

        start = self._visit(node.start_expr)
        stop = self._visit(node.stop_expr)
        self._parallel_depth += 1
        self._depth += 1
        body = "".join(self._visit(child) for child in node.loop_body)
        self._depth -= 1
        self._parallel_depth -= 1
        return (
            f"{self._nindent}Kokkos::parallel_for("
            f"Kokkos::TeamVectorRange(team, {start}, {stop} + 1),\n"
            f"{self._nindent}    [&](const int {node.variable.name}) {{\n"
            f"{body}{self._nindent}}});\n"
            f"{self._nindent}team.team_barrier();\n")

    def assignment_node(self, node) -> str:
        """Let one member make a team-level write to an array.

        Every member of the team executes the body of a hierarchical launch,
        so an array element assigned outside any of the region's parallel
        loops is written by all of them at once. The values agree, but
        concurrent writes to one element are a race in the memory model
        whatever they carry, so the statement is wrapped in
        ``Kokkos::single``. A scalar needs nothing: it is a per-member local,
        and each member writing its own copy races with no one.

        Inside a parallel loop the members already hold disjoint iterations,
        so the write is theirs alone and is left as it is. So is every
        assignment in the two flat launch shapes, whose members are cells
        rather than lanes of one.

        :param node: the assignment in the captured body.
        :type node: :py:class:`psyclone.psyir.nodes.Assignment`

        :returns: the assignment, wrapped in ``Kokkos::single`` and followed
            by a barrier where the team would otherwise race.
        :rtype: str
        """
        if (not self._parallel_loops or self._parallel_depth
                or not isinstance(node.lhs, ArrayReference)):
            return super().assignment_node(node)

        self._depth += 1
        inner = super().assignment_node(node)
        self._depth -= 1
        return (
            f"{self._nindent}Kokkos::single(Kokkos::PerTeam(team), "
            "[&]() {\n"
            f"{inner}{self._nindent}}});\n"
            f"{self._nindent}team.team_barrier();\n")

    def arrayreference_node(self, node: ArrayReference) -> str:
        """Emit an indexed View access with Fortran lower bounds removed.

        This is the one place a subscript of a described array is written, so
        it is the one place the array's declared origin is applied: no path
        through the writer can reach an element of a View without subtracting
        the offset its description carries. That matters more than it reads.
        An array whose origin the region has right and whose subscripts it has
        wrong compiles, links and runs, and returns the wrong answer.

        The subtraction is emitted whenever the offset is non-empty, which
        includes an origin of ``0``: ``u_e(k - 0)`` states the origin the
        access was written against, where ``u_e(k)`` would be
        indistinguishable from a subscript the offset had never reached.

        A :py:class:`KokkosConstant` is a plain C array rather than a View, so
        it is subscripted with brackets; everything else about the access,
        including the origin, is the same.

        :param node: the array reference in the captured body.

        :returns: the equivalent zero-based View access.

        :raises ValueError: if the region described neither a View nor scratch
            for the array, or if it supplied a different number of index
            offsets than the reference has indices.
        """
        try:
            view = self._views[node.name]
        except KeyError as err:
            raise ValueError(
                f"Array '{node.name}' has no Kokkos View "
                "description.") from err

        if len(node.indices) != len(view.index_offsets):
            raise ValueError(
                f"Array '{node.name}' has {len(node.indices)} kernel indices "
                f"but {len(view.index_offsets)} offsets were supplied.")
        indices = []
        for index, offset in zip(node.indices, view.index_offsets):
            expression = self._visit(index)
            if offset:
                expression = f"({expression} - {offset})"
            indices.append(expression)
        indices.extend(view.extra_indices)
        if isinstance(view, KokkosConstant):
            return f"{node.name}[{indices[0]}]"
        return f"{node.name}({', '.join(indices)})"


__all__ = ["KokkosConstant", "KokkosRegion", "KokkosScalar",
           "KokkosScratch", "KokkosView",
           "KokkosWriter", "extent_names", "is_extent", "is_offset"]
