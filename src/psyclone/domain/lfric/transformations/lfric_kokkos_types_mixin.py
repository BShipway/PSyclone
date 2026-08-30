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

"""Resolve what an LFRic symbol is, for the Kokkos region that will carry it.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

The one constraint that follows is that a method reaching a helper of the
sibling mixin ``LFRicKokkosCallMixin`` does so through ``cls``, resolved on
``LFRicKokkosTrans``. Calling such a method directly on either mixin is
therefore not supported.
"""

import re

from psyclone.configuration import Config
from psyclone.psyir.backend.c import CWriter
from psyclone.psyir.backend.kokkos import extent_names, is_extent
from psyclone.psyir.nodes import Literal, Reference
from psyclone.psyir.symbols import (
    ArrayType, ImportInterface, RoutineSymbol, ScalarType,
    UnsupportedFortranType)
from psyclone.psyir.transformations import TransformationError


class LFRicKokkosTypesMixin:
    """Place LFRic kinds, extents and module constants onto the C ABI.

    Every question here is about what a symbol *is*: the C type its kind maps
    to, the extents its declaration gives it, and whether a symbol the body
    reads can be passed by value. Nothing here builds the region or the
    Fortran that calls it; that is ``LFRicKokkosCallMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The C types the Kokkos backend emits, by what LFRic says a kind
    #: actually is. Widths come from PSyclone's own precision map rather than
    #: from a list of kind names, so ``r_tran`` and ``r_bl`` are accepted or
    #: refused on the same evidence as ``r_def``, and a single-precision
    #: ``r_solver`` build is honoured rather than silently promoted.
    #:
    #: There is deliberately no ``BOOLEAN`` entry. The precision map records
    #: ``l_def: 1``, but LFRic defines ``l_def = kind(.false.)``, which
    #: measures 4 bytes; mapping it would emit ``logical(c_bool)`` against a
    #: ``logical(4)`` actual and the model build would fail. A kind this table
    #: does not name -- ``l_def``, a 16-byte ``r_quad``, an undeclared
    #: precision -- fails closed rather than being guessed at.
    _C_TYPES = {
        (ScalarType.Intrinsic.INTEGER, 4): "int",
        (ScalarType.Intrinsic.REAL, 4): "float",
        (ScalarType.Intrinsic.REAL, 8): "double",
    }

    #: A literal of each intrinsic, usable as the argument of
    #: ``storage_size``. Only the intrinsics :py:attr:`_C_TYPES` admits need
    #: an entry, and the probe is built from this rather than special-cased
    #: per kind name.
    _KIND_PROBES = {
        ScalarType.Intrinsic.INTEGER: "1",
        ScalarType.Intrinsic.REAL: "1.0",
    }

    @classmethod
    def _supported_kinds(cls):
        """Name the widths on the ABI, in the order :py:attr:`_C_TYPES` has.

        Derived from the table rather than spelt out, so the two refusal
        messages that quote it cannot drift from what is actually accepted.

        :returns: a phrase such as ``4-byte integer, 4-byte real and 8-byte
            real``.
        :rtype: str
        """
        widths = [f"{width}-byte {intrinsic.name.lower()}"
                  for intrinsic, width in cls._C_TYPES]
        return " and ".join([", ".join(widths[:-1]), widths[-1]])

    @staticmethod
    def _kind_name(symbol):
        """Return the name of a scalar or array element's kind symbol.

        :param symbol: the symbol whose kind is wanted.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the kind parameter's name, such as ``r_solver``, or ``None``
            if the symbol is not of a scalar type or its precision is not
            named by a symbol.
        :rtype: Optional[str]
        """
        datatype = symbol.datatype
        if isinstance(datatype, ArrayType):
            datatype = datatype.elemental_type
        if not isinstance(datatype, ScalarType):
            return None
        precision = datatype.precision
        if not isinstance(precision, Reference):
            return None
        return precision.symbol.name

    @classmethod
    def _c_type(cls, symbol):
        """Return the C type of a symbol, or ``None`` if it has no mapping.

        :param symbol: the symbol to place on the C ABI.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the C type name, such as ``float``, or ``None`` if the
            symbol is not of a scalar or array-of-scalar type or its kind is
            not one :py:attr:`_C_TYPES` maps.
        :rtype: Optional[str]
        """
        datatype = symbol.datatype
        if isinstance(datatype, ArrayType):
            datatype = datatype.elemental_type
        if not isinstance(datatype, ScalarType):
            return None
        return cls._map_kind(datatype.intrinsic, cls._kind_name(symbol))

    @classmethod
    def _map_kind(cls, intrinsic, kind):
        """Return the C type for one LFRic kind name, or ``None``.

        The width comes from the LFRic configuration's precision map rather
        than from the kernel source, which only ever names the kind.

        :param intrinsic: the Fortran intrinsic type the kind qualifies.
        :type intrinsic:
            :py:class:`psyclone.psyir.symbols.ScalarType.Intrinsic`
        :param str kind: the LFRic kind parameter, such as ``r_tran``.

        :returns: the C type name, such as ``float``, or ``None`` if the
            intrinsic and width together are not on the ABI.
        :rtype: Optional[str]
        """
        if kind is None:
            return None
        precision = Config.get().api_conf("lfric").precision_map
        return cls._C_TYPES.get((intrinsic, precision.get(kind)))

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
            kind = cls._kind_name(symbol)
            if kind is not None:
                kinds[kind] = cls._c_type(symbol)
        for literal in schedule.walk(Literal):
            datatype = literal.datatype
            precision = getattr(datatype, "precision", None)
            if not isinstance(precision, Reference):
                continue
            kind = precision.symbol.name
            kinds[kind] = cls._map_kind(datatype.intrinsic, kind)
        return tuple(
            (kind, kinds[kind]) for kind in sorted(kinds)
            if kinds[kind] is not None)

    @staticmethod
    def _extents(symbol):
        """Return the declared extents of an array, in order.

        Each is the symbol's declared upper bound written as C, so a formal
        or a kernel-local array declared ``dimension(max_length,4)`` gives
        ``("max_length", "4")`` and one declared ``dimension(nlayers+1)``
        gives ``("(nlayers + 1)",)``. The routine serves array formals and
        kernel-local arrays alike; both reach the backend as extents that are
        emitted verbatim.

        The lower bound is rendered too, and required to be ``1``. It is not
        used to compute the extent, because it cannot be anything else once
        it has been checked -- writing ``upper - lower + 1`` would be code no
        test could reach. A lower bound the generated View cannot assume away
        is refused instead, since ``KokkosView`` and ``KokkosScratch`` carry
        integer index offsets and every caller supplies 1.

        :param symbol: the array whose shape is wanted.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: one C extent expression per dimension, empty for a scalar.
        :rtype: tuple[str]

        :raises TransformationError: if a dimension carries no declared
            bounds, if its lower bound is not 1, or if its upper bound is not
            an integer expression the Kokkos backend can write as an extent.
        """
        datatype = symbol.datatype
        if not isinstance(datatype, ArrayType):
            return ()
        writer = CWriter()
        extents = []
        for dimension in datatype.shape:
            lower = getattr(dimension, "lower", None)
            upper = getattr(dimension, "upper", None)
            if lower is None or upper is None:
                raise TransformationError(
                    f"LFRicKokkosTrans requires '{symbol.name}' to be "
                    "declared with explicit bounds.")
            # A visitor lowers the tree it is handed, and this one belongs to
            # a live datatype.
            rendered_lower = writer(lower.copy())
            if rendered_lower != "1":
                raise TransformationError(
                    f"LFRicKokkosTrans requires '{symbol.name}' to be "
                    "declared with a lower bound of 1, but found "
                    f"'{rendered_lower}'.")
            extent = writer(upper.copy())
            if not is_extent(extent):
                raise TransformationError(
                    f"LFRicKokkosTrans requires the extents of "
                    f"'{symbol.name}' to be integer expressions over named "
                    f"sizes, but found '{extent}'.")
            extents.append(extent)
        return tuple(extents)

    @classmethod
    def _extent_names(cls, symbol):
        """Return the names an array's extents are sized from.

        An extent is no longer a single name, so a caller asking whether it
        can be evaluated where the region is launched has to ask about every
        name in it. A literal contributes nothing, so a purely fixed-size
        array reports no names at all.

        :param symbol: the array whose extents are to be resolved.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: every name appearing in any of its extents.
        :rtype: set[str]

        :raises TransformationError: as :py:meth:`_extents` does.
        """
        names = set()
        for extent in cls._extents(symbol):
            names |= extent_names(extent)
        return names

    @classmethod
    def _constants(cls, schedule):
        """Return the module constants the kernel body reads.

        These are not kernel arguments, so the PSy layer has to import each
        one and pass it by value into the region.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: ``(name, container, c_type)`` per constant, name-ordered.
        :rtype: list[tuple[str, str, str]]

        :raises TransformationError: as :py:meth:`_describe_constant` does,
            for any constant that has no place on the generated C ABI.
        """
        table = schedule.symbol_table
        local = {symbol.name for symbol in table.argument_list}
        local |= {symbol.name for symbol in table.automatic_datasymbols}
        constants = {}
        for reference in schedule.walk(Reference):
            symbol = reference.symbol
            if symbol.name in local or isinstance(symbol, RoutineSymbol):
                continue
            if symbol.name in constants:
                continue
            constants[symbol.name] = cls._describe_constant(symbol)
        return [constants[name] for name in sorted(constants)]

    @classmethod
    def _describe_constant(cls, symbol):
        """Resolve one non-local symbol onto the generated C ABI.

        :param symbol: the imported symbol the captured body reads.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the symbol's name, the container it is imported from, and
            the C type it is passed by value as.
        :rtype: tuple[str, str, str]

        :raises TransformationError: if the symbol is neither a kernel
            argument nor imported from a module.
        :raises TransformationError: if its type cannot be resolved, which
            means the source of its container is not on the module search
            path.
        :raises TransformationError: if its kind is not one
            :py:attr:`_C_TYPES` maps.
        """
        if not isinstance(symbol.interface, ImportInterface):
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture '{symbol.name}': it is "
                "neither a kernel argument nor imported from a module.")
        container = symbol.interface.container_symbol.name
        try:
            symbol.resolve_type()
        except Exception as err:                 # pylint: disable=W0703
            # The kind is only stated in the module, so guessing here would
            # put a silently wrong type on the C ABI. Say what is missing
            # instead: the caller decides what PSyclone may read.
            raise TransformationError(
                f"LFRicKokkosTrans cannot type '{symbol.name}' without the "
                f"source of '{container}'. Add its directory to PSyclone's "
                f"module search path. ({err})") from err
        c_type = cls._c_type(symbol)
        if c_type is None:
            c_type = cls._declared_c_type(symbol)
        if c_type is None:
            raise TransformationError(
                f"LFRicKokkosTrans cannot pass '{symbol.name}' from "
                f"'{container}' by value: only {cls._supported_kinds()} "
                "scalars have a place on the generated C ABI.")
        return (symbol.name, container, c_type)

    @classmethod
    def _declared_c_type(cls, symbol):
        """Recover a C type from a declaration PSyIR could not model.

        LFRic module constants routinely carry attributes -- ``PROTECTED``,
        an initialiser -- that leave the frontend with an
        :py:class:`UnsupportedFortranType` holding the original text. The kind
        is still stated there, so read it rather than give up; anything with a
        shape is refused, because only scalars are passed by value.

        :param symbol: the imported symbol whose declaration is to be read.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the C type named by the declaration text, or ``None`` if
            the declaration is not one PSyIR failed to model, has a shape, is
            of no intrinsic the ABI carries, or names a kind
            :py:attr:`_C_TYPES` does not map.
        :rtype: Optional[str]
        """
        datatype = getattr(symbol, "datatype", None)
        if not isinstance(datatype, UnsupportedFortranType):
            return None
        attributes = datatype.declaration.split("::")[0]
        if "DIMENSION" in attributes.upper():
            return None
        match = re.match(
            r"\s*(REAL|INTEGER)\s*\(\s*KIND\s*=\s*(\w+)\s*\)",
            attributes, re.IGNORECASE)
        if not match:
            return None
        intrinsic = {
            "real": ScalarType.Intrinsic.REAL,
            "integer": ScalarType.Intrinsic.INTEGER,
        }[match.group(1).lower()]
        return cls._map_kind(intrinsic, match.group(2).lower())


__all__ = ["LFRicKokkosTypesMixin"]
