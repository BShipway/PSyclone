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

A declaration that states no shape at all is the exception, and there is one
of it: a formal declared ``(:)`` or ``(:,:)``, whose extent Fortran takes from
the actual at the call.
:py:meth:`LFRicKokkosBoundsMixin._resolve_assumed_shapes` puts that extent back
into the declaration -- one integer formal per dimension, measured by the PSy
layer and appended to the kernel's own arguments -- so that every reader below
still reads one shape out of one declaration.
:py:meth:`LFRicKokkosBoundsMixin._implicit_extents` names those formals again
and :py:meth:`LFRicKokkosBoundsMixin._implicit_extent_actuals` writes the
``SIZE`` the PSy layer passes for each, the argument mixin appending the two
to the region's signature and to its call.

The sibling mixins are reached through ``cls``, resolved on
``LFRicKokkosTrans``: ``_validate_bounds`` predicts ``cls._lower_sections``
and ``_bounds`` asks ``cls._resolve_constants`` for a declaration written
over a ``parameter``. Calling a method here directly on this mixin is
therefore not supported.
"""

from psyclone.domain.lfric import LFRicTypes
from psyclone.psyir.backend.c import CWriter
from psyclone.psyir.backend.kokkos import extent_names, is_extent
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import (
    BinaryOperation, IntrinsicCall, Literal, Reference)
from psyclone.psyir.symbols import (
    ArgumentInterface, ArrayType, DataSymbol, ScalarType)
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

    #: The tag each scalar carrying an assumed shape's extent is created
    #: under, followed by the formal it measures and the dimension of it. The
    #: tag is what :py:meth:`_implicit_extents` finds them again by: nothing
    #: else tells a formal this transformation added from one the kernel
    #: declared, and the argument mixin has to know which is which to supply
    #: the measurement only for the ones it added.
    _IMPLICIT_EXTENT_TAG = "kokkos-implicit-extent"

    #: What a declaration carrying no bounds carries instead, and where the
    #: shape it does not state is stated. Both are refusals, and the reason
    #: they are refusals is different, so a reader who is told which of the
    #: two this is knows whether to look at the kernel or at its caller.
    #:
    #: An assumed shape reaches a refusal only where
    #: :py:meth:`_resolve_assumed_shapes` could not have measured it, which is
    #: where it is not a kernel argument: a formal's is measured at the call
    #: and put back into its declaration before anything here reads it.
    _SHAPELESS_WORDING = {
        ArrayType.Extent.DEFERRED:
            "a deferred shape, so its size is stated by an ALLOCATE in the "
            "kernel body and not by its declaration",
        ArrayType.Extent.ATTRIBUTE:
            "an assumed shape and is not one of the kernel's arguments, so "
            "there is no call for its size to be stated by",
    }

    #: The wording for the one assumed shape that is a kernel argument and is
    #: still refused. A dummy declared ``dimension(0:)`` is legal Fortran and
    #: takes its extent from the actual and its origin from itself, so the
    #: shape a View would be given would be read out of two places at once --
    #: the very thing :py:meth:`_bounds` exists to prevent. No GungHo kernel
    #: writes one.
    _STATED_ORIGIN_WORDING = (
        "an assumed shape whose lower bound its declaration states, so its "
        "origin and its size would be read from two different places")

    @classmethod
    def _shapeless_wording(cls, dimension):
        """Name the shape a dimension carries in place of declared bounds.

        :param dimension: the entry of the array's shape that has no bounds,
            which is an ``Extent`` where the declaration stated neither bound
            and an ``ArrayBounds`` where it stated the lower one alone.
        :type dimension: Union[
            :py:class:`psyclone.psyir.symbols.ArrayType.Extent`,
            :py:class:`psyclone.psyir.symbols.ArrayType.ArrayBounds`]

        :returns: what the Fortran declared, worded for a refusal.
        :rtype: str
        """
        if isinstance(dimension, ArrayType.ArrayBounds):
            return cls._STATED_ORIGIN_WORDING
        return cls._SHAPELESS_WORDING.get(dimension, f"'{dimension}'")

    @classmethod
    def _resolve_assumed_shapes(cls, schedule):
        """Give every assumed-shape formal an extent it can be sized by.

        A formal declared ``(:)`` states no extent, but its extent is not
        unknown: Fortran takes it from the actual at the call, and the PSy
        layer holds that actual. So the measurement is added to the kernel's
        arguments -- one integer formal per dimension left out, named
        ``<formal>_extent_<dimension>`` -- and written into the declaration in
        place of the extent that is missing. Everything below this reads a
        declared shape as it always did, and the one place that knows the
        shape came from the call is the argument mixin, which passes ``SIZE``
        of the actual for each formal added here.

        This is C2's mechanism turned around. There, a *declared* extent --
        a stencil's per-cell size -- was the wrong length for a View, and a
        scalar carrying the array's storage extent was added beside it and
        substituted into the generated bounds by name. Here the extent is not
        declared at all, so the scalar is written into the declaration itself
        and no substitution is needed; both carry a ``SIZE`` of a PSy-layer
        array as a new region argument, and
        :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosArgumentMixin._region` appends the two sets of actuals in the order
        the two sets of arguments are declared in.

        Only a formal is resolved, and only where its declaration stated no
        bound of any dimension. A local declared ``(:)`` has no call to be
        measured at, and a formal declared ``(0:)`` -- which PSyIR records as
        bounds whose upper is the assumed extent rather than as the extent
        itself -- is left for :py:meth:`_bounds` to refuse; see
        :py:attr:`_STATED_ORIGIN_WORDING`.

        The schedule is mutated in place, and idempotently: a shape resolved
        here is no longer assumed, so a second call over the same schedule
        finds nothing to do. That is what lets :py:meth:`_substitute_bounds`
        call it unconditionally.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        """
        table = schedule.symbol_table
        integer = ScalarType.integer_type()
        for symbol in list(table.argument_list):
            datatype = symbol.datatype
            if not isinstance(datatype, ArrayType):
                continue
            # Every dimension, not any: a declaration stating one bound of one
            # of them -- 'dimension(0:,:)' -- is one whose origin does not
            # come from where its extent would, and _bounds refuses it whole
            # rather than this measuring the half of it that is plain.
            if not all(dimension is ArrayType.Extent.ATTRIBUTE
                       for dimension in datatype.shape):
                continue
            resolved = []
            for index in range(1, len(datatype.shape) + 1):
                extent = table.new_symbol(
                    f"{symbol.name}_extent_{index}",
                    tag=f"{cls._IMPLICIT_EXTENT_TAG}:{symbol.name}:{index}",
                    symbol_type=DataSymbol,
                    datatype=LFRicTypes("LFRicIntegerScalarDataType")(),
                    interface=ArgumentInterface(
                        ArgumentInterface.Access.READ))
                # Appended immediately, because a symbol carrying an argument
                # interface that the argument list does not hold is a table
                # PSyclone reports as inconsistent.
                table.append_argument(extent)
                # The lower bound is the Fortran default rather than the
                # actual's: an assumed-shape dummy is 1-based whatever the
                # array passed to it was declared as, so this is exactly where
                # the origin is not read from the call.
                resolved.append((Literal("1", integer), Reference(extent)))
            symbol.datatype = ArrayType(datatype.elemental_type, resolved)

    @classmethod
    def _implicit_extents(cls, table):
        """Name the extent scalars :py:meth:`_resolve_assumed_shapes` added.

        Read in the order the argument list holds them rather than the order
        they were created in, because that is the order the generated
        signature declares them and the actuals have to be supplied in the
        same one.

        :param table: the kernel's own symbol table, after the resolution.
        :type table: :py:class:`psyclone.psyir.symbols.SymbolTable`

        :returns: per added formal, the formal it measures and the dimension
            of it, in call order.
        :rtype: dict[str, tuple[str, int]]
        """
        prefix = f"{cls._IMPLICIT_EXTENT_TAG}:"
        tags = {symbol.name: tag for tag, symbol in table.tags_dict.items()
                if tag.startswith(prefix)}
        measured = {}
        for symbol in table.argument_list:
            tag = tags.get(symbol.name)
            if tag is None:
                continue
            _, name, dimension = tag.split(":")
            measured[symbol.name] = (name, int(dimension))
        return measured

    @classmethod
    def _implicit_extent_actuals(cls, formals, actuals, table):
        """Measure each assumed shape on the array the PSy layer passes.

        An assumed-shape formal takes its size from the actual, so the size is
        not in the kernel to be read: the declaration says only that there is
        one. It is in the PSy layer, and ``SIZE`` on the actual is the
        question Fortran itself answers when it shapes the dummy. The measured
        value crosses as a scalar formal of the region --
        :py:meth:`_resolve_assumed_shapes` has already made the extent a
        formal and written it into the declaration -- and this supplies its
        actual. It sits beside the resolution rather than with the rest of the
        argument list because the two are one mechanism: the formal is of no
        use without the measurement, and the measurement is of none without
        the formal.

        The array is measured whole, after
        :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosArgumentMixin._per_cell` has unsliced it. That is the same array
        the dummy is shaped from: LFRic slices a trailing cell dimension and
        no other, so dimension *n* of the formal is dimension *n* of the
        actual either way, and the whole array is what the call site can name
        once the launch has replaced the cell loop.

        The route is the one
        :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosArgumentMixin._storage_extents` opened for a stencil's storage: a
        companion scalar formal appended after the kernel's own, carrying
        ``SIZE`` of an actual. The two differ in what they answer rather than
        in how they answer it -- there, an extent the kernel states and a View
        cannot honour; here, an extent the kernel does not state at all -- so
        they measure separate arrays and are kept apart, and both are supplied
        in the order
        :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosArgumentMixin._region_arguments` declares them.

        :param formals: the kernel's own formals, in call order.
        :type formals: list[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        :param actuals: the actuals the PSy layer passes for them, in the same
            order and already unsliced by
            :py:meth:`~psyclone.domain.lfric.transformations.\
LFRicKokkosArgumentMixin._per_cell`.
        :type actuals: list[:py:class:`psyclone.psyir.nodes.DataNode`]
        :param table: the kernel's own table, after the resolution.
        :type table: :py:class:`psyclone.psyir.symbols.SymbolTable`

        :returns: the formals with each measured extent appended to them, and
            the expressions the PSy layer passes for those, in call order.
            The two are returned together because appending to one without
            the other is what would misalign the call.
        :rtype: tuple[list[:py:class:`psyclone.psyir.symbols.DataSymbol`],
            list[:py:class:`psyclone.psyir.nodes.IntrinsicCall`]]
        """
        passed = dict(zip((formal.name for formal in formals), actuals))
        extents = []
        measurements = []
        for name, (measured, dimension) in cls._implicit_extents(
                table).items():
            extents.append(table.lookup(name))
            measurements.append(IntrinsicCall.create(
                IntrinsicCall.Intrinsic.SIZE,
                [Reference(passed[measured].symbol),
                 ("dim", Literal(str(dimension),
                                 ScalarType.integer_type()))]))
        return formals + extents, measurements

    @staticmethod
    def _render(writer, symbol, expression):
        """Return one declared bound written as C.

        The refusal exists because the writer's own failure is a
        ``VisitorError``, which :py:meth:`validate` is not allowed to raise:
        a caller asking whether a kernel can be captured gets an answer or a
        ``TransformationError``, never a back-end exception. A real GungHo
        declaration reaches it -- ``dimension(MAX(nlayers-above,1))`` --
        where the shape is arithmetic the C writer has no intrinsic for.

        :param writer: the C writer rendering the bound.
        :type writer: :py:class:`psyclone.psyir.backend.c.CWriter`
        :param symbol: the array the bound was declared for, named in the
            refusal.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param expression: the bound to render, owned by the caller and
            lowered by this call.
        :type expression: :py:class:`psyclone.psyir.nodes.DataNode`

        :returns: the bound written as C.
        :rtype: str

        :raises TransformationError: if the C writer cannot render it.
        """
        try:
            return writer(expression)
        except VisitorError as err:
            raise TransformationError(
                f"LFRicKokkosTrans cannot write the declared shape of "
                f"'{symbol.name}' as C: {err}") from err

    @staticmethod
    def _span(lower, upper):
        """Return PSyIR counting the elements between two declared bounds.

        The count is ``upper - lower + 1``, and it is built rather than
        rendered so that the one definition serves both
        :py:meth:`_bounds`, which writes it into a View's extent, and
        :py:meth:`_substitute_bounds`, which puts it where the kernel asked
        ``SIZE``. Those two must agree: a View sized by one expression and a
        kernel told another is a wrong answer rather than a refusal.

        A literal lower bound is folded into the upper bound rather than
        subtracted from it, so ``dimension(0:nlayers)`` gives ``nlayers + 1``
        instead of ``nlayers - 0 + 1``. The Fortran default of 1 folds to
        nothing at all, which is what keeps every extent the transformation
        already generated byte for byte what it was. The bounds are not
        evaluated -- ``dimension(2:4)`` gives ``4 - 1`` rather than ``3`` --
        because folding a literal against a literal would need a rule for a
        declared extent that came out negative, and no declaration this
        reaches has one.

        :param lower: the declared lower bound.
        :type lower: :py:class:`psyclone.psyir.nodes.DataNode`
        :param upper: the declared upper bound.
        :type upper: :py:class:`psyclone.psyir.nodes.DataNode`

        :returns: a new tree, owned by the caller, for the element count.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        integer = ScalarType.integer_type()
        if (isinstance(lower, Literal) and
                lower.datatype.intrinsic == ScalarType.Intrinsic.INTEGER):
            shift = 1 - int(lower.value)
            if shift == 0:
                return upper.copy()
            operator = (BinaryOperation.Operator.ADD if shift > 0
                        else BinaryOperation.Operator.SUB)
            return BinaryOperation.create(
                operator, upper.copy(), Literal(str(abs(shift)), integer))
        return BinaryOperation.create(
            BinaryOperation.Operator.ADD,
            BinaryOperation.create(BinaryOperation.Operator.SUB,
                                   upper.copy(), lower.copy()),
            Literal("1", integer))

    @classmethod
    def _bounds(cls, symbol):
        """Return the declared origin and extent of each dimension, in order.

        One declaration is read once, and both answers come out of that
        reading. The origin is the declared lower bound written as C and the
        extent is :py:meth:`_span` of the two bounds written as C, so a
        kernel-local array declared ``dimension(0:nlayers)`` gives
        ``(("0", "(nlayers + 1)"),)`` and one declared
        ``dimension(max_length,4)`` gives ``(("1", "max_length"), ("1",
        "4"))``. The routine serves array formals and kernel-local arrays
        alike; both reach the backend as strings emitted verbatim.

        Reading the two together is the point. The extent sizes the View and
        the origin shifts every subscript of it, and an array whose origin is
        not 1 needs both to move: a View of the right size indexed from the
        wrong place compiles, runs and returns the wrong answer.

        A bound written over a named constant is resolved to what that
        constant was declared as, by the method that resolves the same names
        in the body. ``integer(kind=i_def), parameter :: nfaces = 4`` beside
        ``real(kind=r_tran), dimension(nfaces) :: v_dot_n`` is the commonest
        kernel-local shape in GungHo there is, and the value is in the
        Fortran: an extent left as ``nfaces`` would be refused for naming
        something the launch cannot evaluate, when the launch does not need
        to evaluate it. Only names the declaration has values for are
        replaced, so ``dimension(order+1,nfaces)`` keeps the formal and loses
        the constant.

        :param symbol: the array whose shape is wanted.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the C origin and C extent of each dimension, empty for a
            scalar.
        :rtype: tuple[tuple[str, str]]

        :raises TransformationError: if a dimension carries no declared
            bounds, naming the shape it was declared with instead.
        :raises TransformationError: if a declared bound names something the
            C writer cannot render.
        :raises TransformationError: if its declared lower bound, or the
            extent its two bounds give, is not an integer expression the
            Kokkos backend can write.
        """
        datatype = symbol.datatype
        if not isinstance(datatype, ArrayType):
            return ()
        writer = CWriter()
        bounds = []
        for dimension in datatype.shape:
            lower = getattr(dimension, "lower", None)
            upper = getattr(dimension, "upper", None)
            # A dimension with no bounds at all is the Extent itself; one
            # declared 'dimension(0:)' keeps its lower bound and carries the
            # Extent in place of its upper. Both are shapeless, and the
            # wording tells them apart.
            if lower is None or isinstance(upper, ArrayType.Extent):
                raise TransformationError(
                    f"LFRicKokkosTrans requires '{symbol.name}' to be "
                    f"declared with explicit bounds, but it is declared with "
                    f"{cls._shapeless_wording(dimension)}.")
            # Both bounds are resolved before either is rendered, so that the
            # extent and the origin are read with every name the Fortran has
            # already given a value taken out of them.
            lower = cls._resolve_constants(lower)
            upper = cls._resolve_constants(upper)
            # A visitor lowers the tree it is handed, and this one belongs to
            # a live datatype.
            origin = cls._render(writer, symbol, lower.copy())
            if not is_extent(origin):
                raise TransformationError(
                    f"LFRicKokkosTrans requires the declared origin of "
                    f"'{symbol.name}' to be an integer expression over named "
                    f"sizes, but found '{origin}'.")
            extent = cls._render(writer, symbol, cls._span(lower, upper))
            if not is_extent(extent):
                raise TransformationError(
                    f"LFRicKokkosTrans requires the extents of "
                    f"'{symbol.name}' to be integer expressions over named "
                    f"sizes, but found '{extent}'.")
            bounds.append((origin, extent))
        return tuple(bounds)

    @classmethod
    def _extents(cls, symbol):
        """Return the declared extents of an array, in order.

        Each is the number of elements the declaration gives that dimension,
        written as C, which is the upper bound alone only when the origin is
        the Fortran default of 1. See :py:meth:`_bounds`.

        :param symbol: the array whose shape is wanted.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: one C extent expression per dimension, empty for a scalar.
        :rtype: tuple[str]

        :raises TransformationError: as :py:meth:`_bounds` does.
        """
        return tuple(extent for _, extent in cls._bounds(symbol))

    @classmethod
    def _origins(cls, symbol):
        """Return the declared origin of each dimension of an array, in order.

        These become the index offsets the region subtracts from every
        subscript of the array, so that a Fortran index reaches the
        zero-based View element the declaration says it names. See
        :py:meth:`_bounds`.

        :param symbol: the array whose origins are wanted.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: one C origin expression per dimension, empty for a scalar.
        :rtype: tuple[str]

        :raises TransformationError: as :py:meth:`_bounds` does.
        """
        return tuple(origin for origin, _ in cls._bounds(symbol))

    @classmethod
    def _extent_names(cls, symbol):
        """Return the names an array's declared bounds are built from.

        A bound is no longer a single name, so a caller asking whether it can
        be evaluated where the region is launched has to ask about every name
        in it. A literal contributes nothing, so a purely fixed-size array
        based at 1 reports no names at all.

        Origins are reported alongside extents, and have to be: the origin of
        an array centred on zero names a size just as its extent does, and a
        caller that checked only the extent would pass a region an offset it
        cannot evaluate.

        :param symbol: the array whose bounds are to be resolved.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: every name appearing in any of its origins or extents.
        :rtype: set[str]

        :raises TransformationError: as :py:meth:`_bounds` does.
        """
        names = set()
        for origin, extent in cls._bounds(symbol):
            names |= extent_names(origin) | extent_names(extent)
        return names

    @classmethod
    def _substitute_bounds(cls, schedule):
        """Replace every shape enquiry with the bound its declaration gives.

        ``LBOUND``, ``UBOUND`` and ``SIZE`` are resolved symbolically against
        the symbol table, not evaluated: each call is replaced by a **copy of
        the declared bound's PSyIR**, so the backend renders it by the path it
        renders any other expression and no new writer support is needed.
        ``LBOUND`` gives the declared lower bound rather than the constant 1,
        and ``SIZE`` gives :py:meth:`_span` of the two bounds rather than the
        upper bound alone; the two coincide only for an array based at 1,
        which is why an array based anywhere else answers all three
        differently.

        Every assumed-shape formal is resolved first, by
        :py:meth:`_resolve_assumed_shapes`, so that a kernel asking ``SIZE`` of
        one is answered with the extent measured at the call rather than
        refused for a declaration that states none. The two belong together:
        this is the method that makes a body's shape enquiries agree with the
        shape a View is given, and after the resolution the two are again one
        declaration.

        The schedule is mutated in place, which is why :py:meth:`validate`
        predicts this over a copy rather than running it.

        :param schedule: the kernel schedule to be captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if a shape enquiry's first argument is
            not a plain reference to a declared array, if its dimension is not
            an integer literal within the array's rank, or if a rank-2 or
            higher array is asked for its ``SIZE`` without one.
        :raises TransformationError: as :py:meth:`_bounds` does, unwrapped,
            so a reader gets the bounds grammar's own message rather than a
            paraphrase of it.
        """
        cls._resolve_assumed_shapes(schedule)
        for call in schedule.walk(IntrinsicCall):
            if call.intrinsic not in cls._BOUND_INTRINSICS:
                continue
            symbol, dimension = cls._bound_target(call)
            cls._bounds(symbol)
            # The shape is a live piece of the symbol's datatype, so every
            # branch below copies before it hands a tree to the schedule.
            declared = symbol.datatype.shape[dimension - 1]
            if call.intrinsic is IntrinsicCall.Intrinsic.LBOUND:
                replacement = declared.lower.copy()
            elif call.intrinsic is IntrinsicCall.Intrinsic.UBOUND:
                replacement = declared.upper.copy()
            else:
                replacement = cls._span(declared.lower, declared.upper)
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
