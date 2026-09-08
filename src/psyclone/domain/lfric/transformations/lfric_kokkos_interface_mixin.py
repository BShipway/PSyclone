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

"""Write the ``bind(C)`` interface the PSy layer calls the region through.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

This is the one place the generated Fortran and the generated C++ meet, and
everything here is about that meeting rather than about an argument's own
description, which is ``LFRicKokkosArgumentMixin``'s:

* :py:meth:`LFRicKokkosInterfaceMixin._interface` writes the ``interface``
  block itself -- the wrapped signature, the ``use iso_c_binding`` line and one
  declaration per argument -- and
  :py:meth:`LFRicKokkosInterfaceMixin._launch_symbol` puts it in the PSy
  layer's table as the routine the generated call is made through.
* :py:meth:`LFRicKokkosInterfaceMixin._kind_types` and
  :py:meth:`LFRicKokkosInterfaceMixin._kind_assertions` are the width contract
  the arguments alone do not carry: a local or a literal crosses no interface,
  so nothing outside the generated file constrains its width, and the
  assertions written into the interface body make a rebuild at another
  precision a compile error naming the kind rather than a wrong answer.
* :py:meth:`LFRicKokkosInterfaceMixin._as_c_bool` is the one crossing made by
  conversion rather than by width, so it is the one argument the assertions
  say nothing about.

:py:attr:`LFRicKokkosInterfaceMixin._FORTRAN_TYPES` comes with them, being read
only by the two of them that write Fortran.

It is a module of its own because ``LFRicKokkosArgumentMixin`` answers a
question per *argument* -- what the region declares for it and what the PSy
layer passes -- and these answer one question per *region*, after every
argument has been described. Both halves had grown to where a capability could
not be added to either.

The one constraint that follows is that a method here reaching a helper of the
sibling mixins ``LFRicKokkosTypesMixin`` and ``LFRicKokkosCallMixin`` does so
through ``cls``, resolved on ``LFRicKokkosTrans``:
:py:meth:`LFRicKokkosInterfaceMixin._kind_types` asks ``cls._c_type``,
``cls._kind_name``, ``cls._default_kind_name``, ``cls._map_kind`` and
``cls._kind_argument``, :py:meth:`\
LFRicKokkosInterfaceMixin._kind_assertions` reads ``cls._C_TYPES``,
``cls._KIND_PROBES`` and ``cls._DEFAULT_KINDS``, and :py:meth:`\
LFRicKokkosInterfaceMixin._as_c_bool` calls ``cls._import_constant``. Calling
a method here directly on this mixin is therefore not supported.
"""

import textwrap

from psyclone.psyir.backend.kokkos import KokkosScalar
from psyclone.psyir.nodes import IntrinsicCall, Literal, Reference
from psyclone.psyir.symbols import RoutineSymbol, UnsupportedFortranType


class LFRicKokkosInterfaceMixin:
    """Write the region's ``bind(C)`` interface and its width assertions.

    What the region declares for each argument, and what the PSy layer passes
    for it, is ``LFRicKokkosArgumentMixin``; what a symbol is in C terms is
    ``LFRicKokkosTypesMixin``; the call that replaces the loop is
    ``LFRicKokkosArgumentMixin._call_region``, which reaches
    :py:meth:`_launch_symbol` here for the symbol to call through.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

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

    @classmethod
    def _kind_types(cls, schedule):
        """Return the C type of every kind the captured body names.

        The region's arguments carry their own C types, but its locals and
        its literals cross no interface: nothing outside the generated file
        constrains them, so a kind the backend cannot resolve is silently
        generated at the C writer's default width. This is what stops that.

        A kind :py:meth:`_map_kind` cannot resolve is left out rather than
        refused, because the argument checks have already refused every kind
        that reaches the ABI; what is left is a local or a literal whose width
        the C writer's own default is free to choose.

        A kind named only as a cast target -- the ``r_def`` of
        ``real(x, r_def)``, which no declaration in the body repeats -- is
        collected too. The backend resolves a cast's width through this table,
        so leaving it out would silently write ``(float)`` for a cast the
        Fortran asked to be ``double``: the one case where an unresolved kind
        changes a value rather than only a local's width. A cast naming a kind
        the precision map does not carry is still left out, and still written
        at that default, because there is no width to write instead.

        A declaration naming no kind is collected under the name
        :py:attr:`_DEFAULT_KINDS` gives it, for the same reason and with more
        force: the width it fixes is not written down anywhere at all, so
        leaving it out would be the one case where nothing -- neither this
        table nor the compiler's own argument check -- had looked at it.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: one ``(kind name, C type)`` pair per resolvable kind,
            name-ordered.
        :rtype: tuple[tuple[str, str], ...]
        """
        table = schedule.symbol_table
        kinds = {}
        for symbol in list(table.argument_list) + list(
                table.automatic_datasymbols):
            c_type = cls._c_type(symbol)
            kind = cls._kind_name(symbol)
            if kind is None and c_type is not None:
                # On the ABI with no kind named at all, since _c_type refuses
                # a width stated in place of a name. The region carries it
                # under the name _DEFAULT_KINDS gives it so that the width it
                # fixes can be asserted like any other.
                kind = cls._default_kind_name(symbol)
            if kind is not None:
                kinds[kind] = c_type
        for literal in schedule.walk(Literal):
            datatype = literal.datatype
            precision = getattr(datatype, "precision", None)
            if not isinstance(precision, Reference):
                continue
            kind = precision.symbol.name
            kinds[kind] = cls._map_kind(datatype.intrinsic, kind)
        for reference in schedule.walk(Reference):
            if not cls._kind_argument(reference):
                continue
            # The call's own datatype says which intrinsic the kind qualifies,
            # which the kind name alone does not: i_def and r_def are both
            # just names until the cast around them says integer or real.
            intrinsic = getattr(reference.parent.datatype, "intrinsic", None)
            kind = reference.symbol.name
            c_type = (cls._map_kind(intrinsic, kind)
                      if intrinsic is not None else None)
            if c_type is None:
                # Left out rather than written in as None. A declaration above
                # may already have resolved this kind, and the filter below
                # drops whatever is left None, so writing it in would lose the
                # width that declaration found.
                continue
            kinds[kind] = c_type
        return tuple(
            (kind, kinds[kind]) for kind in sorted(kinds)
            if kinds[kind] is not None)

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

        A kind ``cls._DEFAULT_KINDS`` named because the declaration did not is
        written without the two things a name would otherwise buy: it is left
        out of the ``use`` line, since ``constants_mod`` declares no such kind,
        and its probe is the bare literal rather than a suffixed one, since
        ``1_default_integer`` would name a kind parameter that does not exist
        where ``1`` is the very kind in question. A region whose only asserted
        kind is that one therefore imports nothing from ``constants_mod``
        rather than importing nothing by name.

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
        defaults = {name for _, name in cls._DEFAULT_KINDS.values()}
        names = ", ".join(kind for kind, _ in asserted if kind not in defaults)
        lines = [f"    use constants_mod, only : {names}"] if names else []
        for kind, c_type in asserted:
            probe = cls._KIND_PROBES[intrinsics[c_type]]
            c_kind = cls._FORTRAN_TYPES[c_type][1]
            suffix = "" if kind in defaults else f"_{kind}"
            lines.append(
                f"    integer(kind=merge(4, -1, "
                f"storage_size({probe}{suffix}) == &\n"
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


__all__ = ["LFRicKokkosInterfaceMixin"]
