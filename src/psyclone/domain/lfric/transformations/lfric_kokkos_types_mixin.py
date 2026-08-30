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
from psyclone.psyir.nodes import Call, IntrinsicCall, Literal, Reference
from psyclone.psyir.symbols import (
    ArrayType, DataSymbol, ImportInterface, RoutineSymbol, ScalarType,
    StaticInterface, UnsupportedFortranType)
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

    #: The shape enquiries answered from an array's declaration rather than
    #: from the array itself. A ``Kokkos::View`` does carry an ``extent``, but
    #: asking it would make the generated code depend on a shape the Fortran
    #: has already stated, so :py:meth:`_substitute_bounds` replaces each of
    #: these with the declared bound before the backend sees it.
    _BOUND_INTRINSICS = (IntrinsicCall.Intrinsic.LBOUND,
                         IntrinsicCall.Intrinsic.UBOUND,
                         IntrinsicCall.Intrinsic.SIZE)

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

    @staticmethod
    def _kind_argument(reference):
        """Return whether a reference names a kind rather than reads data.

        ``real(x, r_def)`` puts ``r_def`` into the tree as a
        :py:class:`~psyclone.psyir.nodes.Reference` like any other, but it is
        a type name: the writer consumes it as a cast target and never emits
        it, so there is nothing for the PSy layer to pass by value. The
        frontend names that argument ``kind`` whether or not the Fortran
        spelt ``kind=``, so the test is on the name rather than on the
        position, and it comes from the intrinsic's own argument list rather
        than from a list of intrinsics kept here.

        :param reference: the reference the captured body holds.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`

        :returns: whether this reference is an intrinsic's ``kind`` argument.
        :rtype: bool
        """
        call = reference.parent
        if not isinstance(call, IntrinsicCall):
            return False
        # children[0] is the reference to the intrinsic itself, so the
        # reference's position is one past its index in the argument list.
        index = reference.position - 1
        names = call.argument_names
        if not 0 <= index < len(names):
            return False
        return (names[index] or "").lower() == "kind"

    @staticmethod
    def _called_routine(reference):
        """Return whether a reference names what a call calls.

        A kernel calling a function in a module PSyclone has not read gets a
        plain :py:class:`~psyclone.psyir.symbols.Symbol`, not a
        ``RoutineSymbol``: the frontend cannot tell ``f(i)`` from ``a(i)``
        without the module, and ``resolve_type`` only specialises the symbol
        once something asks. So the test is where the reference sits rather
        than what its symbol has been specialised to.

        :param reference: the reference the captured body holds.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`

        :returns: whether this reference is the routine of a call.
        :rtype: bool
        """
        call = reference.parent
        return isinstance(call, Call) and reference is call.routine

    @staticmethod
    def _static_constant(symbol):
        """Return the literal value a module-level ``parameter`` was given.

        A kernel module routinely declares its own constants -- ``nfaces = 4``,
        ``tol = 1.0e-9_r_def`` -- beside the routine that reads them. These
        are compile-time values, so the region carries the *value* rather than
        an argument: importing the symbol into the PSy layer would not even
        compile, since a kernel module is ``private`` by default and makes
        only its ``_code`` routine public.

        An array ``parameter`` such as ``x_dofs(2) = (/ 1, 3 /)`` has no
        literal to substitute and is not a scalar the ABI could carry either,
        so it is left for :py:meth:`_describe_constant` to refuse.

        :param symbol: the symbol the captured body reads.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`

        :returns: the declared value, or ``None`` if this symbol is not a
            module-level constant declared with a literal.
        :rtype: Optional[:py:class:`psyclone.psyir.nodes.Literal`]
        """
        if not isinstance(symbol, DataSymbol):
            return None
        if not isinstance(symbol.interface, StaticInterface):
            return None
        if not symbol.is_constant:
            return None
        value = symbol.initial_value
        return value if isinstance(value, Literal) else None

    @classmethod
    def _substitute_constants(cls, schedule):
        """Replace each module-level ``parameter`` by the value it was given.

        The companion of :py:meth:`_substitute_bounds`, and for the same
        reason: the Fortran has already stated the value, so the region
        carries it rather than asking for it. :py:meth:`_constants` skips
        exactly what this replaces, so the two cannot disagree about which
        symbols reach the ABI.

        :param schedule: the kernel schedule being captured, modified in
            place.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        """
        for reference in schedule.walk(Reference):
            value = cls._static_constant(reference.symbol)
            if value is None:
                continue
            # The initial value is a live piece of the symbol, as a declared
            # bound is; substituting it without copying would move it out of
            # the symbol table and into the body.
            reference.replace_with(value.copy())

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

        A kind named only as a cast target -- the ``r_def`` of
        ``real(x, r_def)``, which no declaration in the body repeats -- is
        collected too. The backend resolves a cast's width through this table,
        so leaving it out would silently write ``(float)`` for a cast the
        Fortran asked to be ``double``: the one case where an unresolved kind
        changes a value rather than only a local's width. A cast naming a kind
        the precision map does not carry is still left out, and still written
        at that default, because there is no width to write instead.

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
    def _substitute_bounds(cls, schedule):
        """Replace every shape enquiry with the bound its declaration gives.

        ``LBOUND``, ``UBOUND`` and ``SIZE`` are resolved symbolically against
        the symbol table, not evaluated: each call is replaced by a **copy of
        the declared bound's PSyIR**, so the backend renders it by the path it
        renders any other expression and no new writer support is needed.
        ``UBOUND`` and ``SIZE`` give the same node, because
        :py:meth:`_extents` has already required the lower bound to be 1.

        The schedule is mutated in place, which is why :py:meth:`validate`
        predicts this over a copy rather than running it.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if a shape enquiry's first argument is
            not a plain reference to a declared array, if its dimension is not
            an integer literal within the array's rank, or if a rank-2 or
            higher array is asked for its ``SIZE`` without one.
        :raises TransformationError: as :py:meth:`_extents` does, unwrapped,
            so a reader gets the extent grammar's own message rather than a
            paraphrase of it.
        """
        for call in schedule.walk(IntrinsicCall):
            if call.intrinsic not in cls._BOUND_INTRINSICS:
                continue
            symbol, dimension = cls._bound_target(call)
            cls._extents(symbol)
            if call.intrinsic is IntrinsicCall.Intrinsic.LBOUND:
                replacement = Literal("1", ScalarType.integer_type())
            else:
                # The tree is a live piece of the symbol's datatype.
                replacement = symbol.datatype.shape[dimension - 1].upper.copy()
            call.replace_with(replacement)

    @staticmethod
    def _bound_target(call):
        """Return the array a shape enquiry asks about and which dimension.

        :param call: the ``LBOUND``, ``UBOUND`` or ``SIZE`` call.
        :type call: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the declared array and the 1-based dimension asked for.
        :rtype: tuple[:py:class:`psyclone.psyir.symbols.DataSymbol`, int]

        :raises TransformationError: if the first argument is not a plain
            reference to a declared array, rather than an element of one, a
            component of a structure or a symbol of unknown type.
        :raises TransformationError: if the dimension is given by anything
            other than an integer literal, is outside the array's rank, or is
            omitted for anything but ``SIZE`` of a rank-1 array.
        """
        name = call.intrinsic.name
        arguments = call.arguments
        # An ArrayReference is a Reference, and asking one of these about an
        # element rather than the array is a different question, so the test
        # is for the exact type. isinstance would accept the element, and
        # naming the subclasses to exclude would miss the next one added.
        # pylint: disable-next=unidiomatic-typecheck
        if not arguments or type(arguments[0]) is not Reference:
            raise TransformationError(
                f"LFRicKokkosTrans requires the first argument of '{name}' to "
                "be a plain reference to a declared array.")
        symbol = arguments[0].symbol
        if (not isinstance(symbol, DataSymbol) or
                not isinstance(symbol.datatype, ArrayType)):
            raise TransformationError(
                f"LFRicKokkosTrans cannot resolve '{name}' of "
                f"'{symbol.name}', which is not declared as an array.")
        rank = len(symbol.datatype.shape)
        if len(arguments) == 1:
            if call.intrinsic is not IntrinsicCall.Intrinsic.SIZE or rank != 1:
                raise TransformationError(
                    f"LFRicKokkosTrans requires '{name}' of "
                    f"'{symbol.name}' to name a dimension, since "
                    f"'{symbol.name}' is declared with rank {rank}.")
            return symbol, 1
        # Anything past the second argument is 'kind', which asks about the
        # result's type rather than the array's shape.
        named = [given for given in call.argument_names[1:]
                 if given is not None and given.lower() != "dim"]
        dimension = arguments[1]
        if (len(arguments) > 2 or named or
                not isinstance(dimension, Literal) or
                dimension.datatype.intrinsic != ScalarType.Intrinsic.INTEGER):
            raise TransformationError(
                f"LFRicKokkosTrans requires the dimension of '{name}' of "
                f"'{symbol.name}' to be an integer literal.")
        index = int(dimension.value)
        if not 1 <= index <= rank:
            raise TransformationError(
                f"LFRicKokkosTrans cannot resolve '{name}' of '{symbol.name}' "
                f"in dimension {index}, since it is declared with rank "
                f"{rank}.")
        return symbol, index

    @classmethod
    def _constants(cls, schedule):
        """Return the module constants the kernel body reads.

        These are not kernel arguments, so the PSy layer has to import each
        one and pass it by value into the region.

        Three kinds of non-local reference are not data the ABI carries, and
        are skipped rather than described:

        * an intrinsic's ``kind`` argument, which names a type -- see
          :py:meth:`_kind_argument`;
        * the routine of a ``Call``, which names something to call. Before
          ``resolve_type`` is reached these arrive as a plain ``Symbol``
          rather than a ``RoutineSymbol``, so the check is structural;
        * a module-level ``parameter`` declared with a literal value, which
          :py:meth:`_substitute_constants` writes into the body instead.

        The first two are refused by :py:meth:`_validate_body` and by the
        backend well before this, so skipping them here loses no check. What
        it buys is that each refusal names one fact: a kernel calling a
        function was reported both as an uncapturable call and as a module
        constant that could not be passed by value.

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
            if cls._called_routine(reference) or cls._kind_argument(reference):
                continue
            if cls._static_constant(symbol) is not None:
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

        :raises TransformationError: if the symbol is a module-level
            ``parameter`` whose value is not a literal, such as an array.
        :raises TransformationError: if the symbol is neither a kernel
            argument nor imported from a module.
        :raises TransformationError: if its type cannot be resolved, which
            means the source of its container is not on the module search
            path.
        :raises TransformationError: if it turns out to name a routine, which
            only becomes visible once its container has been read.
        :raises TransformationError: if its kind is not one
            :py:attr:`_C_TYPES` maps.
        """
        if not isinstance(symbol.interface, ImportInterface):
            if isinstance(symbol.interface, StaticInterface) and getattr(
                    symbol, "is_constant", False):
                raise TransformationError(
                    f"LFRicKokkosTrans cannot capture '{symbol.name}': a "
                    "module-level constant is written into the region as its "
                    "value, and this one was not declared with a literal "
                    "value the region could carry.")
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
        if isinstance(symbol, RoutineSymbol):
            # resolve_type specialises the symbol, so a name the frontend
            # could only guess at is known to be a routine by now even though
            # _constants could not tell when it walked past it.
            raise TransformationError(
                f"LFRicKokkosTrans cannot capture '{symbol.name}' from "
                f"'{container}': it is a routine rather than data, and a "
                "routine the body calls is refused by the check on calls "
                "rather than passed by value.")
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
