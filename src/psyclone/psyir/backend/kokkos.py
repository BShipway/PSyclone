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
    extents: Tuple[str, ...]
    index_offsets: Tuple[int, ...] = ()
    extra_indices: Tuple[str, ...] = ()
    read_only: bool = False
    random_access: bool = False
    managed: bool = False


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


class KokkosWriter(CWriter):
    """Generate a C++/Kokkos translation unit for a captured region."""

    _SUPPORTED_TYPES = ("double", "float", "int")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._views = {}
        self._kind_types = {}

    def __call__(self, region: KokkosRegion) -> str:
        """Generate code for ``region``.

        :param region: the captured region to generate.

        :returns: a complete C++ translation unit.
        """
        self._validate(region)
        self._views = {
            argument.name: argument for argument in region.arguments
            if isinstance(argument, KokkosView)
        }
        self._kind_types = dict(region.kind_types)

        signature = ",\n    ".join(
            self._argument_declaration(argument)
            for argument in region.arguments)
        views = "\n".join(
            self._view_declaration(argument)
            for argument in region.arguments
            if isinstance(argument, KokkosView))

        self._depth = 2
        local_declarations = "".join(
            self.gen_local_variable(symbol)
            for symbol in region.schedule.symbol_table.automatic_datasymbols)
        body = "".join(
            self._visit(child) for child in region.schedule.children)
        self._depth = 0
        self._views, self._kind_types = {}, {}

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
            "Kokkos::Unmanaged | Kokkos::RandomAccess>;\n\n"
            f"{views}\n\n"
            f'  Kokkos::parallel_for("{region.name}", '
            f"Kokkos::RangePolicy<>(0, {region.cell_count}),\n"
            "      KOKKOS_LAMBDA(const int cell) {\n"
            f"{local_declarations}{body}"
            "      });\n"
            "  Kokkos::fence();\n"
            "}\n")

    @staticmethod
    def _is_identifier(value):
        """Return whether ``value`` is a C++ identifier."""
        return isinstance(value, str) and bool(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))

    def _validate(self, region):
        """Reject incomplete or unsupported region descriptions."""
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

    def _validate_view(self, view):
        """Validate the ownership and dimensional contract for one View."""
        if view.managed:
            raise ValueError(f"Kokkos View '{view.name}' must be unmanaged.")
        if not self._is_identifier(view.data_name):
            raise ValueError(
                f"Kokkos View data name '{view.data_name}' is invalid.")
        if not view.extents or not all(
                self._is_identifier(extent) for extent in view.extents):
            raise ValueError(
                f"Kokkos View '{view.name}' must have named extents.")
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
        """Return one declaration in the generated C ABI."""
        if isinstance(argument, KokkosScalar):
            return f"const {argument.c_type} {argument.name}"
        const = "const " if argument.read_only else ""
        return f"{const}{argument.c_type} *{argument.data_name}"

    @staticmethod
    def _view_declaration(view):
        """Return an unmanaged View declaration."""
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
        """Emit an indexed View access with Fortran lower bounds removed."""
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


__all__ = ["KokkosRegion", "KokkosScalar", "KokkosView", "KokkosWriter"]
