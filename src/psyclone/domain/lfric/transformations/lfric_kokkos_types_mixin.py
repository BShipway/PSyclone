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
    #: There is deliberately no ``BOOLEAN`` entry, and there must not be one:
    #: this table is keyed by width, and a logical is admitted precisely
    #: because its width is never consulted. See :py:attr:`_C_LOGICAL_TYPE`.
    #: A kind this table does not name -- a 16-byte ``r_quad``, an undeclared
    #: precision -- fails closed rather than being guessed at.
    _C_TYPES = {
        (ScalarType.Intrinsic.INTEGER, 4): "int",
        (ScalarType.Intrinsic.REAL, 4): "float",
        (ScalarType.Intrinsic.REAL, 8): "double",
    }

    #: The one C type admitted without consulting the precision map. A logical
    #: crosses this interface by conversion rather than reinterpretation --
    #: the dummy is ``logical(c_bool), value`` and the call site wraps the
    #: actual in ``LOGICAL(..., c_bool)`` -- so the two widths need not agree
    #: and there is no width to assert.
    #:
    #: PSyclone issue #1941 records LFRic's ``l_def`` as 1 byte where it is 4,
    #: and this attribute exists so that entry is never reached rather than
    #: worked around: a corrected #1941 would not change what is generated
    #: here. It is a separate attribute rather than a :py:attr:`_C_TYPES` row
    #: for the same reason -- a row would have to name a width, and both
    #: :py:meth:`_supported_kinds` and
    #: :py:meth:`~psyclone.domain.lfric.transformations.\
    #: lfric_kokkos_call_mixin.LFRicKokkosCallMixin._kind_assertions` read
    #: that table as widths.
    _C_LOGICAL_TYPE = "bool"

    #: A literal of each intrinsic, usable as the argument of
    #: ``storage_size``. Only the intrinsics :py:attr:`_C_TYPES` admits need
    #: an entry, and the probe is built from this rather than special-cased
    #: per kind name.
    _KIND_PROBES = {
        ScalarType.Intrinsic.INTEGER: "1",
        ScalarType.Intrinsic.REAL: "1.0",
    }

    #: What a declaration naming no kind is on the ABI, by intrinsic: the C
    #: type it crosses as, and the name this transformation gives the kind the
    #: declaration did not name.
    #:
    #: A default kind cannot be looked up the way :py:attr:`_C_TYPES` looks up
    #: a named one, because the precision map is keyed by kind name and a
    #: default kind is exactly the one with no name to key it by. Its width is
    #: asserted instead, by the compiler that builds the generated code: the
    #: name here labels a ``storage_size`` check of the bare literal
    #: :py:attr:`_KIND_PROBES` gives against its ``iso_c_binding``
    #: counterpart, so the width below is measured where it is used rather
    #: than assumed here. LFRic's ``constants_mod`` declares no such kind --
    #: being unnamed is the whole of what makes it the default -- so
    #: :py:meth:`~psyclone.domain.lfric.transformations.\
    #: lfric_kokkos_call_mixin.LFRicKokkosCallMixin._kind_assertions` keeps
    #: the name out of the ``use`` line it writes and out of the literal's
    #: kind suffix.
    #:
    #: ``REAL`` has no entry and is refused: LFRic names a kind on every real
    #: it means -- ``r_def``, ``r_solver``, ``r_single`` and ``r_tran`` are all
    #: in use and all different -- so an unkinded real is more likely an
    #: oversight than a default, and the assertion above would only pin down a
    #: width nobody chose. ``LOGICAL`` has no entry either, for the opposite
    #: reason: it is admitted without one, ahead of any width, by
    #: :py:attr:`_C_LOGICAL_TYPE`.
    _DEFAULT_KINDS = {
        ScalarType.Intrinsic.INTEGER: ("int", "default_integer"),
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
        The two clauses after the widths are appended rather than derived,
        because neither is a row of a table keyed by width: a logical is on
        the ABI without one at all (:py:attr:`_C_LOGICAL_TYPE`), and a
        declaration naming no kind is on it under a width the generated code
        asserts rather than looks up (:py:attr:`_DEFAULT_KINDS`).

        :returns: a phrase such as ``4-byte integer, 4-byte real and 8-byte
            real, logical of any kind, and integer or logical declared with no
            kind``.
        :rtype: str
        """
        widths = [f"{width}-byte {intrinsic.name.lower()}"
                  for intrinsic, width in cls._C_TYPES]
        return (" and ".join([", ".join(widths[:-1]), widths[-1]])
                + ", logical of any kind, and integer or logical declared "
                  "with no kind")

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
    def _unnamed_kind(datatype):
        """Return whether a scalar type's kind was left unstated entirely.

        PSyIR records a precision three ways: a
        :py:class:`~psyclone.psyir.nodes.Reference` to a kind parameter, as
        ``integer(kind=i_def)`` gives; a width in bytes, as ``integer*8`` and
        ``real(kind=8)`` give; and ``UNDEFINED``, which is what is left when
        the declaration said nothing. Only the last is a default kind.

        The distinction is the point of the routine. A width is a kind the
        declaration did state, just not by name, so reading it as the default
        would put an ``integer*8`` on the ABI as a C ``int`` and drop four
        bytes of every value without saying so -- the precise failure
        :py:attr:`_DEFAULT_KINDS` exists to assert against.

        :param datatype: the scalar type whose precision is in question.
        :type datatype: :py:class:`psyclone.psyir.symbols.ScalarType`

        :returns: whether the declaration named no kind and stated no width.
        :rtype: bool
        """
        return datatype.precision is ScalarType.Precision.UNDEFINED

    @classmethod
    def _default_kind_name(cls, symbol):
        """Return the name given to the kind a symbol's declaration omits.

        The companion of :py:meth:`_kind_name`, and asked only where that has
        returned ``None``: a symbol declared ``integer, intent(in) :: nlayers``
        names no kind, but the generated body still fixes a width for it, so
        the region carries a kind under the name :py:attr:`_DEFAULT_KINDS`
        gives it and asserts that width like any other.

        A logical is not named here even though it too is admitted without a
        kind. It crosses by conversion and so fixes no width, and a name
        returned here is a name an assertion would be written for; see
        :py:attr:`_C_LOGICAL_TYPE`.

        Asked only of a symbol :py:meth:`_c_type` has already admitted without
        a kind name, which is what makes the question answerable from the
        intrinsic alone: a width stated in place of a name has been refused by
        then, so a symbol reaching here stated neither.

        :param symbol: the symbol whose declaration named no kind.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the name this transformation gives the unnamed kind, such as
            ``default_integer``, or ``None`` if its intrinsic is not one that
            crosses the ABI at a width, a logical being the case in point.
        :rtype: Optional[str]
        """
        datatype = symbol.datatype
        if isinstance(datatype, ArrayType):
            datatype = datatype.elemental_type
        default = cls._DEFAULT_KINDS.get(datatype.intrinsic)
        return default[1] if default else None

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
            symbol is not of a scalar or array-of-scalar type, or its kind is
            not one :py:attr:`_C_TYPES` maps, or its declaration stated a
            width in place of a kind name, or it is a ``logical`` array.
        :rtype: Optional[str]
        """
        datatype = symbol.datatype
        array = isinstance(datatype, ArrayType)
        if array:
            datatype = datatype.elemental_type
        if not isinstance(datatype, ScalarType):
            return None
        boolean = datatype.intrinsic is ScalarType.Intrinsic.BOOLEAN
        if array and boolean:
            # A logical is on the ABI by conversion, which is per value. An
            # array crosses by reference: a View<bool*> over logical(l_def)
            # storage would reinterpret 4-byte elements as 1-byte ones rather
            # than convert them, which is the very failure conversion removes
            # for a scalar. See _C_LOGICAL_TYPE.
            return None
        kind = cls._kind_name(symbol)
        if kind is None and not boolean and not cls._unnamed_kind(datatype):
            # A width stated in place of a kind name. _map_kind reads a missing
            # name as the default kind, which this one is not, so it is not
            # asked. A logical is exempt because it crosses at no width at all.
            return None
        return cls._map_kind(datatype.intrinsic, kind)

    @classmethod
    def _map_kind(cls, intrinsic, kind):
        """Return the C type for one LFRic kind name, or ``None``.

        The width comes from the LFRic configuration's precision map rather
        than from the kernel source, which only ever names the kind.

        :param intrinsic: the Fortran intrinsic type the kind qualifies.
        :type intrinsic:
            :py:class:`psyclone.psyir.symbols.ScalarType.Intrinsic`
        :param kind: the LFRic kind parameter, such as ``r_tran``, or ``None``
            where the declaration named none.
        :type kind: Optional[str]

        :returns: the C type name, such as ``float``, or
            :py:attr:`_C_LOGICAL_TYPE` for any ``logical`` kind, or the type
            :py:attr:`_DEFAULT_KINDS` gives the intrinsic where no kind was
            named, or ``None`` if the intrinsic and width together are not on
            the ABI.
        :rtype: Optional[str]
        """
        if intrinsic is ScalarType.Intrinsic.BOOLEAN:
            # Answered before the precision map is opened, not merely without
            # using the answer: the map's l_def entry is wrong (#1941) and
            # this is what makes that irrelevant rather than survivable. It is
            # also answered before the kind is looked at, which is what admits
            # a plain 'logical': l_def is kind(.false.), so naming it and
            # leaving it out say the same thing, and neither says a width.
            return cls._C_LOGICAL_TYPE
        if kind is None:
            # No name for the precision map to be keyed by, so the width comes
            # from the assertion _DEFAULT_KINDS describes instead of from here.
            default = cls._DEFAULT_KINDS.get(intrinsic)
            return default[0] if default else None
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

        A declaration that names no kind is read too, and handed to
        :py:meth:`_map_kind` with ``None`` so that it is answered by the one
        table the typed path is answered by. What that costs is a regular
        expression that has to tell ``INTEGER, PUBLIC`` from ``INTEGER(i_def)``
        rather than only looking for ``KIND=``: a positional kind is a kind,
        and reading it as the default would put a silently narrowed value on
        the ABI, so the type name is accepted bare only where the declaration
        goes straight on to its attributes or its ``::``.

        :param symbol: the imported symbol whose declaration is to be read.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the C type named by the declaration text, or ``None`` if
            the declaration is not one PSyIR failed to model, has a shape, is
            of no intrinsic the ABI carries, names a kind :py:attr:`_C_TYPES`
            does not map, states one positionally rather than as ``KIND=``, or
            names none where :py:attr:`_DEFAULT_KINDS` admits none.
        :rtype: Optional[str]
        """
        datatype = getattr(symbol, "datatype", None)
        if not isinstance(datatype, UnsupportedFortranType):
            return None
        attributes = datatype.declaration.split("::")[0]
        if "DIMENSION" in attributes.upper():
            return None
        match = re.match(
            r"\s*(REAL|INTEGER|LOGICAL)\s*"
            r"(?:\(\s*KIND\s*=\s*(\w+)\s*\)|(?=,|$))",
            attributes, re.IGNORECASE)
        if not match:
            return None
        intrinsic = {
            "real": ScalarType.Intrinsic.REAL,
            "integer": ScalarType.Intrinsic.INTEGER,
            "logical": ScalarType.Intrinsic.BOOLEAN,
        }[match.group(1).lower()]
        kind = match.group(2)
        return cls._map_kind(intrinsic, kind.lower() if kind else None)


__all__ = ["LFRicKokkosTypesMixin"]
