# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""A deliberately small Kokkos backend for whole LFRic loop regions."""

from dataclasses import dataclass
import re
from typing import Tuple, Union

from psyclone.psyir.backend.c import CWriter
from psyclone.psyir.nodes import (
    ArrayReference, CodeBlock, KernelSchedule, Reference)
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
    index_offsets: Tuple[int, ...] = ()
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
    index_offsets: Tuple[int, ...] = ()
    extra_indices: Tuple[str, ...] = ()


@dataclass(frozen=True)
class KokkosRegion:
    """All information required to generate one Kokkos translation unit."""

    name: str
    schedule: KernelSchedule
    cell_count: str
    # Spelt out rather than given a module-level alias: ``autoapi`` renders
    # every module variable as a literal block and Sphinx then appends its
    # own "alias of" line unindented, which fails the ``-W`` doc build.
    arguments: Tuple[Union[KokkosScalar, KokkosView], ...]
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


class KokkosWriter(CWriter):
    """Generate a C++/Kokkos translation unit for a captured region."""

    _SUPPORTED_TYPES = ("double", "float", "int")

    def __init__(self, **kwargs):
        """Create a writer holding no region.

        :param kwargs: additional keyword arguments for
            :py:class:`~psyclone.psyir.backend.c.CWriter`.
        :type kwargs: unwrapped dict
        """
        super().__init__(**kwargs)
        self._views = {}
        self._kind_types = {}

    def __call__(self, region: KokkosRegion) -> str:
        """Generate code for ``region``.

        The region's :py:attr:`KokkosRegion.kind_types` are in force for the
        duration of the call and cleared afterwards, so that a writer reused
        for a second region does not carry the first one's widths into it.

        One of two launch shapes is generated, selected by whether the region
        describes any :py:attr:`KokkosRegion.scratch`. A region without
        scratch is generated exactly as it was before scratch existed; see
        :py:meth:`_range_launch` and :py:meth:`_team_launch`.

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
        self._kind_types = dict(region.kind_types)

        signature = ",\n    ".join(
            self._argument_declaration(argument)
            for argument in region.arguments)
        views = "\n".join(
            self._view_declaration(argument)
            for argument in region.arguments
            if isinstance(argument, KokkosView))
        team_aliases = "".join(
            f"  {alias}\n" for alias in (
                "using TeamPolicy = Kokkos::TeamPolicy<>;",
                "using TeamMember = TeamPolicy::member_type;",
                "using ScratchSpace = "
                "Kokkos::DefaultExecutionSpace::scratch_memory_space;",
            )) if region.scratch else ""

        # The team shape nests the body one level deeper, inside the
        # TeamThreadRange lambda.
        scratch_names = {item.name for item in region.scratch}
        self._depth = 3 if region.scratch else 2
        local_declarations = "".join(
            self.gen_local_variable(symbol)
            for symbol in region.schedule.symbol_table.automatic_datasymbols
            # A scratch array is declared as a View over team scratch, so its
            # symbol must not also be declared here. ``gen_declaration``
            # renders an array local as ``double * restrict x_new`` -- a
            # pointer to nothing, which compiles and would shadow the View.
            if symbol.name not in scratch_names)
        body = "".join(
            self._visit(child) for child in region.schedule.children)
        self._depth = 0
        self._views, self._kind_types = {}, {}

        launch = (
            self._team_launch(region, local_declarations, body)
            if region.scratch
            else self._range_launch(region, local_declarations, body))

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
    def _range_launch(region, local_declarations, body):
        """Return the ``RangePolicy`` launch, one cell per iteration.

        This is the shape every region had before scratch existed, and it is
        reproduced here unchanged: the captures already in the model are gated
        on whole-model checksums and on assertions over this exact text.

        :param region: the region being generated.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
        :param local_declarations: the generated declarations of the kernel's
            scalar locals, already indented.
        :type local_declarations: str
        :param body: the generated kernel body, already indented.
        :type body: str

        :returns: the ``parallel_for`` and its captured body.
        :rtype: str
        """
        return (
            f'  Kokkos::parallel_for("{region.name}", '
            f"Kokkos::RangePolicy<>(0, {region.cell_count}),\n"
            "      KOKKOS_LAMBDA(const int cell) {\n"
            f"{local_declarations}{body}"
            "      });\n")

    @staticmethod
    def _team_launch(region, local_declarations, body):
        """Return the ``TeamPolicy`` launch, one cell per team rank.

        Cells are tiled across the ranks of a team so that each rank takes one
        cell and holds its own per-thread scratch. That keeps the parallelism
        identical to :py:meth:`_range_launch` -- one cell per worker -- while
        giving each worker fast, launch-scoped storage; on a GPU that scratch
        is shared memory rather than global.

        The team size cannot be chosen here, because it depends on how much
        scratch each rank asks for, so the policy is asked for the largest it
        supports. The scratch request is set on the probe policy before the
        query, or the answer is the one for a policy requesting nothing.

        :param region: the region being generated.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
        :param local_declarations: the generated declarations of the kernel's
            scalar locals, already indented.
        :type local_declarations: str
        :param body: the generated kernel body, already indented.
        :type body: str

        :returns: the scratch type aliases, the size computation, the bound
            body, the team-size probe and the ``parallel_for``.
        :rtype: str
        """
        aliases = "".join(
            f"  using {item.name}_scratch_t = Kokkos::View<{item.c_type}"
            f"{'*' * len(item.extents)}, Kokkos::LayoutLeft, ScratchSpace, "
            "Unmanaged>;\n"
            for item in region.scratch)
        sizes = "\n      + ".join(
            f"{item.name}_scratch_t::shmem_size({', '.join(item.extents)})"
            for item in region.scratch)
        constructions = "".join(
            f"      {item.name}_scratch_t {item.name}("
            f"team.thread_scratch(0), {', '.join(item.extents)});\n"
            for item in region.scratch)
        return (
            f"{aliases}\n"
            f"  const size_t scratch_bytes = {sizes};\n\n"
            "  auto body = KOKKOS_LAMBDA(const TeamMember &team) {\n"
            "    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, "
            "team.team_size()),\n"
            "        [&](const int rank) {\n"
            "      const int cell = team.league_rank() * team.team_size() "
            "+ rank;\n"
            # The league is sized by rounding up, so the last team runs with
            # ranks that have no cell. Without this they would run the body
            # for a cell past the end of every View.
            f"      if (cell >= {region.cell_count}) {{\n"
            "        return;\n"
            "      }\n"
            f"{constructions}"
            f"{local_declarations}{body}"
            "    });\n"
            "  };\n\n"
            "  TeamPolicy probe = TeamPolicy(1, Kokkos::AUTO)\n"
            "      .set_scratch_size(0, Kokkos::PerThread(scratch_bytes));\n"
            "  const int team_size = probe.team_size_max(body, "
            "Kokkos::ParallelForTag());\n"
            f"  const int league_size = ({region.cell_count} + team_size - 1)"
            " / team_size;\n"
            f'  Kokkos::parallel_for("{region.name}",\n'
            "      TeamPolicy(league_size, team_size)\n"
            "          .set_scratch_size(0, "
            "Kokkos::PerThread(scratch_bytes)),\n"
            "      body);\n")

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
            in :py:attr:`_SUPPORTED_TYPES`, or if a View's index offsets are
            not integers.
        :raises ValueError: if the region's name, its cell count, an argument
            name, a kind name or a View's data name or region indices are not
            C++ identifiers; if a View's or a scratch array's extent is not an
            integer expression over named sizes; if the schedule contains a
            :py:class:`~psyclone.psyir.nodes.CodeBlock`; if two arguments
            share a C ABI name; if the cell count is not itself a scalar
            argument; if a kernel argument has no description; if a View
            breaks the ownership or dimensional contract
            :py:meth:`_validate_view` states; or if a scratch array breaks the
            contract :py:meth:`_validate_scratch` states.
        """
        # pylint: disable=too-many-branches
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
            :py:attr:`_SUPPORTED_TYPES`, or its index offsets are not
            integers.
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
            for name in extent_names(extent):
                if name not in scalar_names:
                    raise ValueError(
                        f"Kokkos scratch '{scratch.name}' has extent "
                        f"'{extent}', which is sized from '{name}' rather "
                        "than from a scalar argument.")
        if len(scratch.extents) != len(scratch.index_offsets):
            raise ValueError(
                f"Kokkos scratch '{scratch.name}' dimensions do not match its "
                "kernel indices.")
        if not all(isinstance(offset, int)
                   for offset in scratch.index_offsets):
            raise TypeError(
                f"Kokkos scratch '{scratch.name}' index offsets must be "
                "integers.")

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
        :raises TypeError: if its index offsets are not integers.
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
        if not all(isinstance(offset, int) for offset in view.index_offsets):
            raise TypeError(
                f"Kokkos View '{view.name}' index offsets must be integers.")
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

    def literal_node(self, node) -> str:
        """Write a literal at the width its own Fortran kind has.

        ``2.0_r_solver`` is a ``float`` in a single-precision build, and C++
        would otherwise read the generated ``2.0`` as a ``double`` and promote
        the whole expression around it.

        :param node: the literal to write.
        :type node: :py:class:`psyclone.psyir.nodes.Literal`

        :returns: the C representation of the literal.
        :rtype: str
        """
        text = super().literal_node(node)
        if self._kind_c_type(node.datatype) == "float":
            return f"{text}f"
        return text

    def arrayreference_node(self, node: ArrayReference) -> str:
        """Emit an indexed View access with Fortran lower bounds removed.

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
        return f"{node.name}({', '.join(indices)})"


__all__ = ["KokkosRegion", "KokkosScalar", "KokkosScratch", "KokkosView",
           "KokkosWriter", "extent_names", "is_extent"]
