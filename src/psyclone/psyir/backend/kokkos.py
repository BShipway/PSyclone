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
from psyclone.psyir.nodes import ArrayReference, CodeBlock, KernelSchedule


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


class KokkosWriter(CWriter):
    """Generate a C++/Kokkos translation unit for a captured region."""

    _SUPPORTED_TYPES = ("double", "int")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._views = {}

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
        self._views = {}

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
