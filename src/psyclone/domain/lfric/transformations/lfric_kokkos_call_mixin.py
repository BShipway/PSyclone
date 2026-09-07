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

"""Describe the Kokkos region, and the Fortran the PSy layer calls it through.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

The one constraint that follows is that a method reaching a helper of the
sibling mixins ``LFRicKokkosTypesMixin`` and ``LFRicKokkosBoundsMixin`` does
so through ``cls``, resolved on ``LFRicKokkosTrans``. Calling such a method
directly on any of them is therefore not supported, and several methods here
do reach across: :py:meth:`LFRicKokkosCallMixin._region_arguments` and
:py:meth:`LFRicKokkosCallMixin._local_arrays` both ask ``cls._c_type`` and
``cls._extents``, and :py:meth:`LFRicKokkosCallMixin._kind_assertions` reads
``cls._C_TYPES`` and ``cls._KIND_PROBES``.
"""

import textwrap

from psyclone.psyir.backend.kokkos import (
    KokkosScalar, KokkosScratch, KokkosView)
from psyclone.psyir.nodes import Loop, Reference
from psyclone.psyir.symbols import (
    ArgumentInterface, ContainerSymbol, DataSymbol, ImportInterface,
    RoutineSymbol, UnresolvedType, UnsupportedFortranType)


class LFRicKokkosCallMixin:
    """Build the generated region's signature and the call that reaches it.

    Every question here is about the *interface* between the PSy layer and the
    generated C++: what the region is called, what arguments it takes, what
    scratch it needs, and what ``bind(C)`` interface block declares it. What a
    symbol is, in C terms, is ``LFRicKokkosTypesMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The region's iteration count, and the second extent of every per-cell
    #: array. Named by the PSy layer, not by the kernel.
    _CELL_COUNT = "ncells"

    #: Per C type, the Fortran declaration the ``bind(C)`` interface uses and
    #: the ``iso_c_binding`` kind that declaration needs imported. One table
    #: rather than two, so the interface's ``use`` line and its declarations
    #: cannot disagree.
    #: ``bool`` is first so that the ``use iso_c_binding`` line an interface
    #: writes stays in this table's order whichever types it carries.
    _FORTRAN_TYPES = {
        "bool": ("logical(c_bool)", "c_bool"),
        "int": ("integer(c_int)", "c_int"),
        "float": ("real(c_float)", "c_float"),
        "double": ("real(c_double)", "c_double"),
    }

    @staticmethod
    def _drop_unused_counter(routine, symbol):
        """Undeclare the replaced loop's counter if nothing else counts by it.

        The PSy layer declares one counter per iteration space, so an invoke
        whose only cell loop is captured is left declaring a variable it
        never mentions again -- which LFRic compiles with
        ``-Werror=unused-variable``. An invoke with a second cell loop keeps
        it, which is why this asks rather than assumes.

        :param routine: the PSy-layer routine holding the replaced loop.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param symbol: the counter the replaced loop was counting with.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        table = routine.symbol_table
        if table.lookup(symbol.name, otherwise=None) is not symbol:
            return
        if symbol in table.argument_list:
            return
        if any(reference.symbol is symbol
               for reference in routine.walk(Reference)):
            return
        # A Loop holds its control variable as an attribute rather than as a
        # Reference, so the walk above would not see a sibling loop still
        # counting with it.
        if any(loop.variable is symbol for loop in routine.walk(Loop)):
            return
        table.remove(symbol)

    @staticmethod
    def _region_name(schedule):
        """Name the generated region after the implementation it captures.

        After the *implementation*, not after the kernel: a kind-polymorphic
        kernel has one name and several implementations, so naming the region
        ``kernel.name`` would give both members of one interface the same
        region name with different C types. Two invokes at different precisions
        in one build would then collide. For a kernel that is not polymorphic
        the schedule carries the kernel's own name, so nothing else moves.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: the region's name, the implementation's with its ``_code``
            component dropped and ``_kokkos`` appended.
        :rtype: str
        """
        name = schedule.name.lower()
        if name.endswith("_code"):
            name = name[:-len("_code")]
        elif "_code_" in name:
            # LFRic names an interface's members '<kernel>_code_<kind>'.
            name = name.replace("_code_", "_", 1)
        return f"{name}_kokkos"

    @classmethod
    def _region_arguments(cls, formals, per_cell, constants, cell_index):
        """Describe the generated signature for the backend.

        The formals are passed in rather than read from the schedule because
        one of them may already have been dropped: a kernel taking an LMA
        operator has a leading cell argument the region declares instead of
        taking, and :py:meth:`apply` removes it from the formals and the
        actuals together, so that the two stay index-aligned.

        :param formals: the kernel formals the generated signature carries,
            in call order.
        :type formals: list[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        :param set[str] per_cell: formals the PSy layer slices by cell.
        :param constants: the module constants passed by value, as
            :py:meth:`_constants` returns them.
        :type constants: list[tuple[str, str, str]]
        :param str cell_index: the name the launch gives its own cell index,
            which every sliced View is indexed by. It is the region's
            :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_index`
            and is passed rather than assumed because the kernel may declare
            ``cell`` itself.

        :returns: one description per generated C argument, in call order.
        :rtype: tuple[Union[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScalar`,
            :py:class:`psyclone.psyir.backend.kokkos.KokkosView`], ...]
        """
        arguments = []
        for symbol in formals:
            c_type = cls._c_type(symbol)
            extents = cls._extents(symbol)
            if not extents:
                arguments.append(KokkosScalar(symbol.name, c_type))
                continue
            sliced = symbol.name in per_cell
            read_only = (
                symbol.interface.access == ArgumentInterface.Access.READ)
            arguments.append(KokkosView(
                symbol.name, f"{symbol.name}_data", c_type,
                extents + ((cls._CELL_COUNT,) if sliced else ()),
                index_offsets=(1,) * len(extents),
                extra_indices=(cell_index,) if sliced else (),
                read_only=read_only, random_access=read_only))
        arguments.append(KokkosScalar(cls._CELL_COUNT, "int"))
        arguments.extend(
            KokkosScalar(name, c_type) for name, _, c_type in constants)
        return tuple(arguments)

    @classmethod
    def _local_arrays(cls, schedule):
        """Describe the kernel's automatic arrays as team scratch.

        Each one becomes a scratch View private to the team rank running the
        cell, which is what makes the region's per-cell temporaries per-cell.
        A scalar local needs none of this and is declared in the region body,
        so only arrays appear here.

        :py:meth:`_validate_locals` has already refused anything this could
        not describe.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: one description per automatic array, in declaration order.
        :rtype: tuple[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScratch`, ...]
        """
        scratch = []
        for symbol in schedule.symbol_table.automatic_datasymbols:
            if not symbol.is_array:
                continue
            extents = cls._extents(symbol)
            scratch.append(KokkosScratch(
                symbol.name, cls._c_type(symbol), extents,
                index_offsets=(1,) * len(extents)))
        return tuple(scratch)

    @staticmethod
    def _import_constant(symbol_table, name, container):
        """Return the PSy-layer import for one kernel module constant.

        :param symbol_table: the PSy-layer table the import is added to.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        :param str name: the constant's name in its own module.
        :param str container: the module it is imported from.

        :returns: the existing symbol if the PSy layer already has one, and a
            new imported symbol otherwise.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        existing = symbol_table.lookup(name, otherwise=None)
        if existing:
            return existing
        module = symbol_table.find_or_create(
            container, symbol_type=ContainerSymbol)
        return symbol_table.find_or_create(
            name, symbol_type=DataSymbol, datatype=UnresolvedType(),
            interface=ImportInterface(module))

    @classmethod
    def _launch_symbol(cls, symbol_table, region):
        """Create or return the explicit interoperable launch interface.

        :param symbol_table: the PSy-layer table the interface is added to.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        :param region: the captured region the interface declares.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

        :returns: the symbol the generated call is made through, carrying the
            ``interface`` block as an
            :py:class:`~psyclone.psyir.symbols.UnsupportedFortranType`.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        existing = symbol_table.lookup(region.name, otherwise=None)
        if existing:
            return existing
        symbol = RoutineSymbol(
            region.name, UnsupportedFortranType(cls._interface(region)))
        symbol_table.add(symbol)
        return symbol

    @classmethod
    def _kind_assertions(cls, kind_types):
        """Write the compile-time width checks for one region's kinds.

        The compiler already checks the arguments, because the interface names
        an ``iso_c_binding`` kind where the PSy layer names an LFRic one. It
        cannot check what the generated body assumed about a local or a
        literal, and it cannot check anything at all if the two kinds happen to
        agree today and stop agreeing when LFRic is rebuilt at another
        precision. These assertions close both gaps in the one place the
        generated Fortran and the generated C++ meet.

        Each is the standard Fortran static assert: ``merge`` selects the kind
        ``4`` when the widths match and ``-1`` when they do not, and ``-1`` is
        not a supported integer kind, so a mismatch is a hard compile error
        naming the parameter and therefore the kind.

        :param kind_types: one ``(kind name, C type)`` pair per kind, as
            :py:meth:`_kind_types` returns them.
        :type kind_types: tuple[tuple[str, str], ...]

        :returns: the ``use`` line and one assertion per pair, each line
            already indented for an interface body, or the empty string when
            there are no kinds to assert.
        :rtype: str
        """
        intrinsics = {c_type: intrinsic
                      for (intrinsic, _), c_type in cls._C_TYPES.items()}
        # A logical kind has no width to assert -- it crosses the ABI by
        # conversion, as LFRicKokkosTypesMixin._C_LOGICAL_TYPE explains -- so
        # it is dropped before anything is written, the `use` line included. A
        # region whose only body kind is logical therefore emits no assertion
        # block at all rather than an empty one.
        asserted = [(kind, c_type) for kind, c_type in kind_types
                    if c_type in intrinsics]
        if not asserted:
            return ""
        names = ", ".join(kind for kind, _ in asserted)
        lines = [f"    use constants_mod, only : {names}"]
        for kind, c_type in asserted:
            probe = cls._KIND_PROBES[intrinsics[c_type]]
            c_kind = cls._FORTRAN_TYPES[c_type][1]
            lines.append(
                f"    integer(kind=merge(4, -1, "
                f"storage_size({probe}_{kind}) == &\n"
                f"        storage_size({probe}_{c_kind}))), parameter :: "
                f"assert_kind_{kind} = 0")
        return "\n".join(lines) + "\n"

    @classmethod
    def _interface(cls, region):
        """Write the ``bind(C)`` interface the PSy layer calls through.

        :param region: the captured region the interface declares.
        :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

        :returns: a complete ``interface`` block, for the PSy layer to carry
            as an :py:class:`~psyclone.psyir.symbols.UnsupportedFortranType`.
        :rtype: str
        """
        names = [argument.name for argument in region.arguments]
        signature = textwrap.wrap(
            ", ".join(names), width=58, break_long_words=False)
        header = f"  subroutine {region.name}({signature[0]}"
        for line in signature[1:]:
            header += " &\n      " + line
        declarations = []
        # The assertions name an iso_c_binding kind too, and a body-only kind
        # can have a C type no argument carries, so both sources are counted.
        used = {argument.c_type for argument in region.arguments}
        used |= {c_type for _, c_type in region.kind_types}
        for argument in region.arguments:
            fortran = cls._FORTRAN_TYPES[argument.c_type][0]
            if isinstance(argument, KokkosScalar):
                declarations.append(f"    {fortran}, value :: {argument.name}")
            else:
                intent = "in" if argument.read_only else "inout"
                declarations.append(
                    f"    {fortran}, dimension(*), intent({intent}) :: "
                    f"{argument.name}")
        # Only the kinds this region's arguments declare, in table order, so
        # that a region using none of a kind does not import it unused.
        kinds = ", ".join(kind for c_type, (_, kind)
                          in cls._FORTRAN_TYPES.items() if c_type in used)
        body = "\n".join(declarations)
        # A `use` must precede every other specification statement, so the
        # assertions follow both of them; and they sit inside the interface
        # body so that the generated interface stays self-contained and needs
        # nothing added to the PSy layer around it.
        assertions = cls._kind_assertions(region.kind_types)
        return (
            "interface\n"
            f"{header}) bind(C)\n"
            f"    use iso_c_binding, only : {kinds}\n"
            f"{assertions}"
            f"{body}\n"
            f"  end subroutine {region.name}\n"
            "end interface")


__all__ = ["LFRicKokkosCallMixin"]
