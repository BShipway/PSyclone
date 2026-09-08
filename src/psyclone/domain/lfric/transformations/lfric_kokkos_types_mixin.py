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

What a *declaration* says an array's shape is -- its extents, its declared
lower bounds and the shape enquiries answered from them -- is not here but in
``LFRicKokkosBoundsMixin``, so that one reading of one declaration serves both
the size a View is given and the origin its subscripts are shifted from. Which
of the names a body reads is module state, and what the region does with each,
is ``LFRicKokkosConstantsMixin``; this module answers only what a kind maps to.
Which kinds one captured body names, and the compile-time width assertions
they become, are ``LFRicKokkosInterfaceMixin``, beside the interface that
carries them.

The one constraint that follows is that a method reaching a helper of the
sibling mixins ``LFRicKokkosBoundsMixin``, ``LFRicKokkosArgumentMixin`` or
``LFRicKokkosConstantsMixin`` does so through ``cls``, resolved on
``LFRicKokkosTrans``. Calling such a method directly on either mixin is
therefore not supported.
"""

from psyclone.configuration import Config
from psyclone.psyir.nodes import Call, IntrinsicCall, Reference
from psyclone.psyir.symbols import ArrayType, ScalarType


class LFRicKokkosTypesMixin:
    """Place LFRic kinds onto the C ABI.

    Every question here is about what a symbol *is*: the C type its kind maps
    to, and the widths the generated region has to assert. The shape its
    declaration gives it is ``LFRicKokkosBoundsMixin``; whether the body may
    read it at all, and as what, is ``LFRicKokkosConstantsMixin``. Nothing
    here builds the region, which is ``LFRicKokkosArgumentMixin``, or the
    Fortran that calls it, which is ``LFRicKokkosInterfaceMixin``.
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
    #: lfric_kokkos_interface_mixin.LFRicKokkosInterfaceMixin._kind_assertions`
    #: read that table as widths.
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
    #: lfric_kokkos_interface_mixin.LFRicKokkosInterfaceMixin._kind_assertions`
    #: keeps the name out of the ``use`` line it writes and out of the
    #: literal's kind suffix.
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

    @classmethod
    def _view_intrinsics(cls):
        """Name the Fortran intrinsics a View's elements may have.

        Derived from :py:attr:`_C_TYPES` rather than listed, so that widening
        that table reaches the refusal quoting this -- :py:meth:`\
~psyclone.domain.lfric.transformations.lfric_kokkos_contract_mixin.\
LFRicKokkosContractMixin._validate_field_type` -- without a second edit. It
        is the intrinsics that are asked for and not the widths, because a
        field's declaration states its own kind and that kind is checked
        where every other formal's is, by :py:meth:`\
~psyclone.domain.lfric.transformations.lfric_kokkos_contract_mixin.\
LFRicKokkosContractMixin._validate_formals`.

        ``logical`` is absent, and must stay absent while
        :py:attr:`_C_LOGICAL_TYPE` is what puts a logical on the ABI: it
        crosses by conversion, which is per value, so an array of them is
        refused rather than reinterpreted.

        :returns: the intrinsic names, lower-cased and in alphabetical order,
            as ``('integer', 'real')``.
        :rtype: tuple[str, ...]
        """
        return tuple(sorted(
            {intrinsic.name.lower() for intrinsic, _ in cls._C_TYPES}))

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


__all__ = ["LFRicKokkosTypesMixin"]
