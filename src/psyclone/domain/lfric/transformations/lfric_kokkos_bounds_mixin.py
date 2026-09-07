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

"""Answer what an array's declaration says its shape is.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

One declaration is read here and nowhere else, so that the extent a View is
sized by and the origin its subscripts are shifted from are the same reading of
the same Fortran. :py:meth:`LFRicKokkosBoundsMixin._extents` gives the sizes,
:py:meth:`LFRicKokkosBoundsMixin._extent_names` the names they are built from,
and :py:meth:`LFRicKokkosBoundsMixin._substitute_bounds` rewrites the shape
enquiries ``LBOUND``, ``UBOUND`` and ``SIZE`` into the declared bounds
themselves, using :py:meth:`LFRicKokkosBoundsMixin._bound_target` to say which
array each enquiry asks about.
:py:meth:`LFRicKokkosBoundsMixin._validate_bounds` is the side-effect-free
predicate the capture contract and the coverage survey ask, and it predicts
that rewrite over a copy.

The sibling mixins are reached through ``cls``, resolved on
``LFRicKokkosTrans``: ``_validate_bounds`` predicts ``cls._lower_sections``.
Calling a method here directly on this mixin is therefore not supported.
"""

from psyclone.psyir.backend.c import CWriter
from psyclone.psyir.backend.kokkos import extent_names, is_extent
from psyclone.psyir.nodes import IntrinsicCall, Literal, Reference
from psyclone.psyir.symbols import ArrayType, DataSymbol, ScalarType
from psyclone.psyir.transformations import TransformationError


class LFRicKokkosBoundsMixin:
    """Read an array's declared bounds, and refuse a shape it cannot write.

    Every question here is about the shape a declaration states: the extents
    a ``Kokkos::View`` is sized by, and the shape enquiries the kernel body
    asks that the declaration can answer instead. What a symbol's *kind* maps
    to in C is ``LFRicKokkosTypesMixin``; whether a loop may be captured at
    all is ``LFRicKokkosContractMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The shape enquiries answered from an array's declaration rather than
    #: from the array itself. A ``Kokkos::View`` does carry an ``extent``, but
    #: asking it would make the generated code depend on a shape the Fortran
    #: has already stated, so :py:meth:`_substitute_bounds` replaces each of
    #: these with the declared bound before the backend sees it.
    _BOUND_INTRINSICS = (IntrinsicCall.Intrinsic.LBOUND,
                         IntrinsicCall.Intrinsic.UBOUND,
                         IntrinsicCall.Intrinsic.SIZE)

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
    def _validate_bounds(cls, schedule):
        """Check that every shape enquiry resolves to a declared bound.

        Predicts :py:meth:`_substitute_bounds` over a copy, as
        :py:meth:`_validate_sections` predicts the lowering, because that
        substitution mutates the schedule and :py:meth:`validate` must leave
        it as it found it.

        The copy is lowered first. ``ArrayAssignment2LoopsTrans`` is a
        *producer* of ``LBOUND`` and ``UBOUND``, writing them into the loop
        bounds of every full-extent section it rewrites, so checking before
        the lowering would miss the calls the transformation itself creates.
        On a schedule that is already lowered -- which is what the coverage
        survey hands this method -- the lowering is a no-op, so the one
        method serves both callers. The cost is a second schedule copy per
        validation, taken so that each predicate stays readable alone.

        A schedule whose sections cannot be lowered says nothing about its
        bounds that :py:meth:`_validate_sections` has not already said, so
        that failure is passed over rather than re-reported. Reaching it means
        this method was called on its own, as the coverage survey calls each
        predicate independently; from :py:meth:`validate` the section check
        has refused the schedule before this runs.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if a shape enquiry cannot be resolved
            from the declaration, for any of the reasons
            :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosTypesMixin._substitute_bounds` gives.
        """
        probe = schedule.copy()
        try:
            cls._lower_sections(probe)
        except TransformationError:
            return
        try:
            cls._substitute_bounds(probe)
        except TransformationError as err:
            raise TransformationError(
                "LFRicKokkosTrans cannot resolve an array bound from its "
                f"declaration: {err}") from err


__all__ = ["LFRicKokkosBoundsMixin"]
