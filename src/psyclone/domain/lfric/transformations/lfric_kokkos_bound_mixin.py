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

"""Inline a kernel's calls in whatever order succeeds, and bound a callee's
locals that a written argument would otherwise size.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``classmethod`` or a ``staticmethod``. It carries the inlining loop that
:py:class:`~psyclone.domain.lfric.transformations.lfric_kokkos_inline_mixin.\
LFRicKokkosInlineMixin` used to, because that module stands at pylint's
thousand-line limit and the two rules this adds are about the *order* calls
are inlined in and the *shape* a callee's local is given, which are one
subject.

**Why the order matters.** :py:class:`~psyclone.psyir.transformations.\
InlineTrans` refuses a callee whose local array is sized by a dummy whose
actual "may be written to before the call". Two helpers called inside one
loop, each passing the same extent, are each other's prior access: the
second's call still stands when the first is judged, and a call whose
interface the frontend has not read is taken to write every argument. Whichever
is inlined first is refused for the other's sake. Inlining the *other* first
turns that call into statements whose accesses are plain reads, and the first
then passes; and once the first attempt has brought its callee into the
Container, the second call's judgement can read that callee's intents, which
:py:meth:`~psyclone.psyir.nodes.Reference.is_write` consults. So a refusal is
not final while another call is still pending: every pending call is tried,
the first that inlines is taken, the body is re-walked, and only a pass in
which nothing inlines is a refusal -- reported in the words the first call was
refused with.

**Why a bound is needed.** The same rule refuses a callee whose local is
sized by a dummy whose actual the caller genuinely assigned -- LFRic's
vertical FFSL transport computes ``array_length`` and passes it as a helper's
``nlayers``, which sizes eleven of the helper's locals. Inlined, those locals
would be declared at the top of the caller with a size not yet assigned,
which is the case the rule exists for. The Kokkos back-end sizes a kernel's
locals as team scratch on the host, before the launch, from the region's
scalar arguments, so what it needs is not the exact size but an upper bound
in those scalars. The ``bounded_locals`` option supplies one: for a named
callee and one of its dummies, an expression in the *caller's* scope that
bounds every actual the dummy can take. The callee gains a dummy, the locals
the original dummy sized are sized by the new one instead, and every call
passes the bound as its extra actual. The body is untouched -- its loops
still run to the exact size -- so the only change is that the local is as
long as the bound rather than exactly as long as it needs to be, which is
what a scratch allocation is anyway.

**A bound is a name or an expression.** A name is looked up in the caller's
table, which is what the vertical FFSL transport needs: its helper's column
is bounded by the kernel's own ``nlayers``. Where the caller holds no single
name that bounds the dummy, the bound is written as Fortran over the names it
does hold and parsed against the caller's table. LFRic's horizontal FFSL
transport at cubed-sphere panel edges is that case:
``ffsl_flux_xy_special_edge_code`` sets ``recon_size`` to ``3 + 2*order`` in
one branch and to ``1 + 2*order`` in the other, and passes it to
``ffsl_flux_xy_special_edge_1d``, which sizes ``field_local`` and
``field_local_tmp`` by it. Neither value is a name the caller holds, and
``3 + 2*order`` -- the larger of the two, ``order`` being a reconstruction
order and not negative -- is an expression over one. The sibling kernel this
one was written from, ``ffsl_flux_xy_panel_remap_kernel_mod``, declares the
same local as ``field_local(nlayers, 1+2*order)`` in the helper itself and is
captured for that reason; the bound restores the shape the special-edge
variant lost by hoisting the computation into its caller.

The expression is parsed against a table holding the caller's own symbols, so
a name the caller does not declare is refused rather than invented and
nothing is added to the kernel's table by the attempt. It is held to
arithmetic over literals and plain scalar names, for the reason
:py:mod:`psyclone.psyir.backend.kokkos_spread_extent` holds a spread extent
to them: the launch sizes the scratch these locals become before the functor
runs, and a call or a subscript is not a value it has there. Whether the
region can size scratch from the shape that results is
``LFRicKokkosContractMixin._validate_locals``'s to say, and it says it of
this shape as it does of any other.

That is an assertion made on the kernel's behalf, as the SKIP table and the
capture profile's other tables are: PSyclone cannot prove the bound, and a
bound that is too small is an out-of-bounds write the Fortran build would
have caught with ``-fcheck=bounds`` and the region will not. The option is
therefore validated for its shape here, refused by name where a callee or
dummy it names is not found, and left to the capture profile to state with
its reason beside it.
"""
from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.nodes import (
    ArrayReference, BinaryOperation, Container, IntrinsicCall, Literal, Node,
    Range, Reference, Routine, UnaryOperation)
from psyclone.psyir.symbols import (
    ArgumentInterface, ArrayType, DataSymbol, SymbolTable)
from psyclone.psyir.transformations import InlineTrans, TransformationError


class LFRicKokkosBoundMixin:
    """Inline calls in an order that succeeds, bounding locals where asked.

    **What is accepted.** Any order of inlining that leaves no call standing
    within :py:attr:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_inline_mixin.LFRicKokkosInlineMixin._INLINE_LIMIT` passes; and,
    under the ``bounded_locals`` option, a callee whose automatic local is
    sized by a dummy whose actual is assigned before the call, provided the
    option names that callee, that dummy, and a bound the caller's scope
    holds -- a name of that scope, or an expression over names of it.

    **What is refused.** A pass in which no pending call inlines, with the
    first refusal's own words; an option that is not a mapping of callee
    name to a mapping of dummy name to bound name; an option naming a dummy
    the callee does not take, or a bound that is neither a name the caller
    holds nor a Fortran expression over such names; and a bound expression
    reading anything but literals and plain scalar names, which the launch
    could not evaluate where it sizes the scratch.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The option naming, per callee, the dummies whose declarations are
    #: sized by an upper bound the caller supplies:
    #: ``{"subgrid_quadratic_recon": {"nlayers": "nlayers"}}`` says that the
    #: locals of ``subgrid_quadratic_recon`` that its dummy ``nlayers`` sizes
    #: are sized instead by the caller's ``nlayers``, whatever the call
    #: passes for the dummy itself.
    _BOUNDED_LOCALS_OPTION = "bounded_locals"

    #: The suffix a callee's new bound dummy is named with, after the dummy
    #: it bounds. ``nlayers`` gains ``nlayers_bound``; a clash is resolved
    #: by the symbol table as any new symbol's is.
    _BOUND_SUFFIX = "_bound"

    #: The node types a bound expression may be built from, matched by exact
    #: class rather than by ``isinstance`` for the reason
    #: :py:data:`psyclone.psyir.backend.kokkos_spread_extent._EXTENT_NODES`
    #: is: an ``ArrayReference`` is a ``Reference`` and subscripts storage the
    #: launch does not hold where it sizes the region's scratch. Anything
    #: outside arithmetic over literals and plain scalar names -- a call, a
    #: subscript, a ``CodeBlock`` -- is refused by name.
    _BOUND_NODES = (BinaryOperation, Literal, Reference, UnaryOperation)

    #: The words of the one refusal another inlining can clear:
    #: :py:class:`~psyclone.psyir.transformations.InlineTrans`'s rule that a
    #: declaration depends on an argument "which is passed by argument and"
    #: is or may be written before the call. A call refused for anything
    #: else is refused for good, and trying the other calls first would only
    #: repeat the inliner's work; it is raised at once, as it always was.
    _DEFERRABLE_REFUSAL = "is passed by argument and"

    @classmethod
    def _bounded_locals(cls, options):
        """Read and check the ``bounded_locals`` option.

        :param options: the transformation's options, or None.
        :type options: Optional[Dict[str, Any]]

        :returns: the callee name to dummy name to bound name mapping, every
            name lower-cased; empty where the option is absent.
        :rtype: Dict[str, Dict[str, str]]

        :raises TransformationError: if the option is present and is not a
            mapping of callee names to mappings of dummy names to bound
            names, every one a non-empty string.
        """
        table = (options or {}).get(cls._BOUNDED_LOCALS_OPTION)
        if table is None:
            return {}
        message = (f"LFRicKokkosTrans' '{cls._BOUNDED_LOCALS_OPTION}' option "
                   f"must map a callee name to a mapping of dummy name to "
                   f"bound name, all non-empty strings, but found "
                   f"'{table}'.")
        if not isinstance(table, dict):
            raise TransformationError(message)
        result = {}
        for callee, bounds in table.items():
            if (not isinstance(callee, str) or not callee
                    or not isinstance(bounds, dict) or not bounds):
                raise TransformationError(message)
            entries = {}
            for dummy, bound in bounds.items():
                if (not isinstance(dummy, str) or not dummy
                        or not isinstance(bound, str) or not bound):
                    raise TransformationError(message)
                entries[dummy.lower()] = bound.lower()
            result[callee.lower()] = entries
        return result

    @staticmethod
    def _bound_expressions(symbol):
        """Yield every explicit bound expression of an automatic array.

        A scalar yields nothing; so does a dimension whose extent is deferred
        or assumed, which has no expression to read or rewrite.

        :param symbol: a symbol of the callee's table.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`

        :returns: the lower and upper bound expressions, dimension by
            dimension.
        :rtype: Iterator[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        datatype = getattr(symbol, "datatype", None)
        if not isinstance(datatype, ArrayType):
            return
        for dimension in datatype.shape:
            if isinstance(dimension, ArrayType.ArrayBounds):
                yield dimension.lower
                yield dimension.upper

    @classmethod
    def _sized_by(cls, symbol, dummy):
        """Whether ``symbol`` is an automatic array whose shape uses ``dummy``.

        :param symbol: a symbol of the callee's table.
        :type symbol: :py:class:`psyclone.psyir.symbols.Symbol`
        :param dummy: the dummy argument whose use in a shape is looked for.
        :type dummy: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: True if the symbol's shape references the dummy.
        :rtype: bool
        """
        return any(reference.symbol is dummy
                   for extent in cls._bound_expressions(symbol)
                   for reference in extent.walk(Reference))

    @classmethod
    def _section_uses(cls, routine, symbol):
        """Make every whole-array use of ``symbol`` an explicit section.

        A whole-array reference to a local -- ``cp`` in ``cp = a + b``, or
        ``cp(:)`` -- means the local's declared extent, and once the local
        is re-declared by the caller's bound that extent is the caller's.
        The callee wrote ``cp`` meaning its own ``n`` elements, so each such
        use is pinned before the declaration moves: it becomes ``cp(1:n)``
        with the bounds the callee declared, which the inliner then rewrites
        in terms of the actual as it does every other use of the dummy.
        Skipped, this rewrite left the vertical FFSL helper's whole-array
        statements running over the kernel's ``nlayers`` elements while their
        operands had the column's ``array_length`` (phase 7, 2026-09-13).

        An inquiry -- ``SIZE(cp)``, ``UBOUND(cp)`` -- reads the section's
        bounds instead of the array's, which is the same answer when the
        lower bound is one and a different one otherwise; the latter is
        refused rather than answered wrongly.

        :param routine: the callee, whose body is rewritten in place.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param symbol: the automatic array about to be re-declared.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :raises TransformationError: if the local is the subject of an
            inquiry intrinsic and a lower bound of its is not one.
        """
        shape = symbol.datatype.shape
        unit_lower = all(
            isinstance(dimension.lower, Literal)
            and dimension.lower.value == "1" for dimension in shape)
        uses = [reference for reference in routine.walk(Reference)
                if reference.symbol is symbol]
        for reference in uses:
            parent = reference.parent
            if (not unit_lower and isinstance(parent, IntrinsicCall)
                    and parent.is_inquiry):
                raise TransformationError(
                    f"LFRicKokkosTrans cannot bound the local '{symbol.name}' "
                    f"of '{routine.name}': it is the subject of "
                    f"{parent.intrinsic.name} and its lower bound is not 1, "
                    f"so a section would answer that inquiry differently.")
            if type(reference) is Reference:  # pylint: disable=C0123
                reference.replace_with(ArrayReference.create(symbol, [
                    Range.create(dimension.lower.copy(),
                                 dimension.upper.copy())
                    for dimension in shape]))
                continue
            # An array symbol is otherwise referenced with indices; a full
            # range among them is spelt with the array's own bounds, which
            # are about to move.
            for index, child in enumerate(getattr(reference, "indices", ())):
                if isinstance(child, Range) and reference.is_full_range(index):
                    child.children[0].replace_with(shape[index].lower.copy())
                    child.children[1].replace_with(shape[index].upper.copy())

    @classmethod
    def _rebind_shape(cls, symbol, dummy, bound):
        """Give ``symbol`` its shape with ``dummy`` replaced by ``bound``.

        The datatype is copied and the copy edited, so that a shape shared
        with any other symbol -- which the frontend does not produce, but
        which nothing forbids -- is not edited under it.

        :param symbol: the automatic array to re-declare.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param dummy: the dummy every reference to which is replaced.
        :type dummy: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param bound: the bound dummy that replaces it.
        :type bound: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        symbol.datatype = symbol.datatype.copy()
        for extent in cls._bound_expressions(symbol):
            for reference in extent.walk(Reference):
                if reference.symbol is dummy:
                    reference.symbol = bound

    @classmethod
    def _bound_locals(cls, call, table):
        """Size the callee's locals the option names by the caller's bound.

        For each dummy the option names for ``call``'s callee: a bound dummy
        is added to the callee (once, however many calls reach it), every
        automatic local whose shape the dummy sized is re-declared with the
        bound dummy in its place, and the call gains the caller's bound as an
        extra actual. Made on the callee as the call site sees it, after
        :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_inline_mixin.LFRicKokkosInlineMixin._module_inline`, so that
        :py:class:`~psyclone.psyir.transformations.InlineTrans` then finds a
        declaration that depends on an argument nobody writes.

        :param call: the call whose callee is to be prepared.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param table: the option, as :py:meth:`_bounded_locals` returns it.
        :type table: Dict[str, Dict[str, str]]

        :raises TransformationError: if the option names a dummy the callee
            does not take, or a bound the caller's scope does not hold.
        """
        # pylint: disable=too-many-locals
        # The method the option is read by is on the mixin this one sits
        # beside; the composed class resolves both.
        # pylint: disable=no-member
        entries = table.get(cls._callee_name(call).lower())
        if not entries:
            return
        caller = call.ancestor(Routine)
        for routine in cls._local_callees(call):
            routine_table = routine.symbol_table
            for dummy_name, bound_name in entries.items():
                dummy = routine_table.lookup(dummy_name, otherwise=None)
                if dummy is None or dummy not in routine_table.argument_list:
                    raise TransformationError(
                        f"LFRicKokkosTrans' '{cls._BOUNDED_LOCALS_OPTION}' "
                        f"option names '{dummy_name}' as a dummy of "
                        f"'{routine.name}', which takes no such argument.")
                actual = cls._bound_actual(
                    caller, routine.name, dummy_name, bound_name)
                bound = routine_table.lookup(
                    f"{dummy_name}{cls._BOUND_SUFFIX}", otherwise=None)
                if bound is None:
                    # The argument list is read before the new symbol gains
                    # its interface: the table refuses to list arguments
                    # while a symbol with an argument interface is not among
                    # them, which is the state between the two steps.
                    arguments = list(routine_table.argument_list)
                    bound = routine_table.new_symbol(
                        root_name=f"{dummy_name}{cls._BOUND_SUFFIX}",
                        symbol_type=DataSymbol,
                        datatype=dummy.datatype.copy())
                    bound.interface = ArgumentInterface(
                        ArgumentInterface.Access.READ)
                    routine_table.specify_argument_list(arguments + [bound])
                    for symbol in routine_table.automatic_datasymbols:
                        if cls._sized_by(symbol, dummy):
                            cls._section_uses(routine, symbol)
                            cls._rebind_shape(symbol, dummy, bound)
                # The actual, once per call: a second call to a callee
                # already widened finds the dummy there and adds only this.
                if len(call.arguments) < len(routine_table.argument_list):
                    call.append_named_arg(None, actual)

    @classmethod
    def _bound_actual(cls, caller, callee_name, dummy_name, bound_name):
        """Return the actual a call passes for a bounded dummy.

        A bound that is a name of the caller's scope becomes a
        :py:class:`~psyclone.psyir.nodes.Reference` to it, which is what the
        vertical FFSL transport's ``nlayers`` is. Anything else is read as a
        Fortran expression over the caller's names by
        :py:meth:`_bound_expression`, which is what the horizontal
        special-edge transport's ``3 + 2*order`` needs.

        A name the caller's scope does not hold is not read as an expression
        and then refused for naming itself: it is refused here, in the words
        the option's reader has always used, so that a misspelt bound reads
        as a misspelt bound.

        :param caller: the routine the call is made from, or ``None`` where
            the call is not in one.
        :type caller: Optional[:py:class:`psyclone.psyir.nodes.Routine`]
        :param str callee_name: the callee's name, for the message.
        :param str dummy_name: the dummy being bounded, for the message.
        :param str bound_name: the bound the option gives, a name or an
            expression.

        :returns: the expression the call passes as the bound's actual.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TransformationError: if the bound is a name the caller's
            scope does not hold, or the call is in no routine at all.
        """
        table = caller.symbol_table if caller is not None else None
        if table is not None:
            if not bound_name.isidentifier():
                return cls._bound_expression(
                    caller, callee_name, dummy_name, bound_name)
            target = table.lookup(bound_name, otherwise=None)
            if target is not None:
                return Reference(target)
        raise TransformationError(
            f"LFRicKokkosTrans' '{cls._BOUNDED_LOCALS_OPTION}' option bounds "
            f"'{dummy_name}' of '{callee_name}' by '{bound_name}', which is "
            f"not in scope at the call in "
            f"'{caller.name if caller else '?'}'.")

    @classmethod
    def _bound_expression(cls, caller, callee_name, dummy_name, bound_name):
        """Read a bound written as Fortran over the caller's own names.

        The expression is parsed against a table of the caller's symbols
        rather than against the caller's table itself. The frontend answers a
        name it cannot find by declaring it, and declaring it in the kernel's
        own table would leave a misspelt bound as a symbol of the kernel
        instead of as a refusal. Here such a name lands in a table thrown
        away with the attempt, and the references that reach it are what names
        the refusal. The symbols added are the caller's own objects, not
        copies, so the expression the call carries reads the caller's
        variables and not lookalikes of them.

        The shape of the expression is judged before the names in it. An
        intrinsic call carries a reference to its own name, which no symbol
        table holds, and reporting ``max(n, 2)`` as a bound naming an unknown
        variable ``MAX`` would send the reader looking for a declaration
        rather than for a simpler bound.

        :param caller: the routine the call is made from.
        :type caller: :py:class:`psyclone.psyir.nodes.Routine`
        :param str callee_name: the callee's name, for the message.
        :param str dummy_name: the dummy being bounded, for the message.
        :param str bound_name: the bound the option gives, as Fortran.

        :returns: the parsed expression, reading the caller's own symbols.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TransformationError: if the text is not a Fortran expression,
            if it is not arithmetic over literals and plain scalar names, or
            if it names anything the caller's scope does not hold.
        """
        message = (f"LFRicKokkosTrans' '{cls._BOUNDED_LOCALS_OPTION}' option "
                   f"bounds '{dummy_name}' of '{callee_name}' by "
                   f"'{bound_name}'")
        scope = SymbolTable()
        for symbol in caller.symbol_table.symbols:
            scope.add(symbol)
        try:
            expression = FortranReader().psyir_from_expression(
                bound_name, scope)
        # The frontend reports a text it cannot parse as an expression with
        # whatever its parser raises; every one of them is a refusal about
        # the option rather than a failure of the capture.
        except Exception as err:            # pylint: disable=broad-except
            raise TransformationError(
                f"{message}, which is not a Fortran expression: "
                f"{type(err).__name__}: {err}") from err
        for node in expression.walk(Node):
            if node.__class__ not in cls._BOUND_NODES:
                raise TransformationError(
                    f"{message}, which the launch could not evaluate where "
                    f"it sizes the region's scratch: a bound is arithmetic "
                    f"over literals and plain scalar names, and this one "
                    f"holds {type(node).__name__} "
                    f"('{node.debug_string().strip()}').")
        known = {id(symbol) for symbol in caller.symbol_table.symbols}
        unknown = sorted({reference.symbol.name
                          for reference in expression.walk(Reference)
                          if id(reference.symbol) not in known})
        if unknown:
            raise TransformationError(
                f"{message}, which names {', '.join(unknown)}: not in scope "
                f"at the call in '{caller.name}'.")
        return expression

    @classmethod
    def _inline_calls(cls, schedule, options=None):
        """Replace every call in ``schedule`` with the callee's statements.

        The body is re-walked after each inlining rather than the calls being
        collected once, because inlining rewrites the tree the collection was
        taken from: a call the callee itself makes did not exist when the walk
        ran, and a call sharing a statement with the one just inlined may have
        been moved out of it.

        Each callee is prepared before it is inlined: brought into the
        Container the call is made from by
        :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_inline_mixin.LFRicKokkosInlineMixin._module_inline`, relaxed
        of a ``TARGET`` formal and of an aliasing ``POINTER`` local, agreed
        with the caller about the origin of a disputed name, read for the
        declarations of what it imports, given a bound for a local a written
        argument would size where the option asks, by :py:meth:`_bound_locals`,
        and handed the one-element section of any element actual it reads as
        an array of one, by
        :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_element_mixin.LFRicKokkosElementMixin._section_element_actuals`.

        A call refused for a declaration depending on a written argument is
        not final while another is pending: the refusal is kept, the next
        pending call is tried, and a pass in which none inlines raises the
        refusal the pass began with. Every other refusal is final and is
        raised at once. See the module docstring for why two helpers in one
        loop need the deferral, and :py:attr:`_DEFERRABLE_REFUSAL` for why
        only that refusal earns it.

        :param schedule: the kernel schedule to rewrite in place.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param options: the transformation's options, read for
            ``bounded_locals``.
        :type options: Optional[Dict[str, Any]]

        :raises TransformationError: if no pending call can be inlined, in
            which case the reason is the one PSyclone gave for the first of
            them, followed by the reason its callee was not brought into the
            Container where there is one. Anything
            :py:class:`~psyclone.psyir.transformations.InlineTrans` raises is
            a reason, not only what it refuses with; a class other than
            ``TransformationError`` is named in the message. A refusal a
            preparation step raises is reported in its own words.
        :raises TransformationError: if calls are still left after
            :py:attr:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_inline_mixin.LFRicKokkosInlineMixin._INLINE_LIMIT` of them have
            been inlined, which is what a self-recursive callee looks like
            from here.
        """
        # pylint: disable=no-member
        table = cls._bounded_locals(options)
        for _ in range(cls._INLINE_LIMIT):
            pending = cls._pending_calls(schedule)
            if not pending:
                return
            refusals = []
            for call in pending:
                try:
                    cls._inline_one(schedule, call, table)
                except TransformationError as err:
                    if cls._DEFERRABLE_REFUSAL not in str(err):
                        raise
                    refusals.append(err)
                    continue
                break
            else:
                raise refusals[0]
        names = ", ".join(sorted(
            {cls._callee_name(call)
             for call in cls._pending_calls(schedule)}))
        raise TransformationError(
            f"LFRicKokkosTrans inlined {cls._INLINE_LIMIT} calls into "
            f"'{schedule.name}' and found the call to {names} still there: "
            f"a routine that calls itself is never inlined away, and no "
            f"kernel this is meant for calls that deeply.")

    @classmethod
    def _already_local(cls, call):
        """Whether the callee of ``call`` is already declared in its Container.

        True once :py:meth:`~psyclone.domain.lfric.transformations.\
lfric_kokkos_inline_mixin.LFRicKokkosInlineMixin._module_inline` has brought
        the callee in -- as a routine, or as a generic interface whose
        specifics travelled with it -- and for a callee the kernel's own
        module declares. False for a callee the Container still only imports,
        or names without declaring.

        :param call: the call whose callee is looked for.
        :type call: :py:class:`psyclone.psyir.nodes.Call`

        :returns: True if the Container declares the callee's name itself.
        :rtype: bool
        """
        # pylint: disable=no-member
        container = call.ancestor(Container)
        if container is None:
            return False
        symbol = container.symbol_table.lookup(
            cls._callee_name(call), otherwise=None)
        return symbol is not None and not symbol.is_import

    @classmethod
    def _inline_one(cls, schedule, call, table):
        """Prepare ``call``'s callee and inline the call, or refuse.

        :param schedule: the kernel schedule the call is in, for the message.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param call: the call to inline.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param table: the ``bounded_locals`` option, as read.
        :type table: Dict[str, Dict[str, str]]

        :raises TransformationError: if the call cannot be inlined, with the
            inliner's reason and, where the callee could not be brought into
            the Container first, that reason after it; or if a preparation
            step fails, in that step's own words.
        """
        # pylint: disable=no-member
        name = cls._callee_name(call)
        # A call tried once and deferred has already had its callee brought
        # in; asking again would add the callee's symbol a second time, which
        # KernelModuleInlineTrans reports with a KeyError rather than a
        # refusal (a generic interface's name, in the survey of rhs_alg_mod).
        # Bringing the callee in is attempted, never required: whatever it
        # raises is kept as the reason to report if the inlining then fails.
        refusal = None
        if not cls._already_local(call):
            try:
                refusal = cls._module_inline(call)
            except Exception as err:  # pylint: disable=broad-except
                refusal = f"{type(err).__name__}: {err}"
        try:
            cls._relax_target_arguments(call)
            cls._alias_locals(call)
            cls._agree_on_imports(call)
            cls._read_declarations(call)
            cls._bound_locals(call, table)
            cls._section_element_actuals(call)
        except TransformationError:
            raise
        # A preparation step that fails in any other way is a refusal too, in
        # its own class's name, rather than a crash a deferred retry turns the
        # whole capture into.
        except Exception as err:  # pylint: disable=broad-except
            raise TransformationError(
                f"LFRicKokkosTrans cannot prepare the call to '{name}' in "
                f"'{schedule.name}' for inlining: {type(err).__name__}: "
                f"{err}") from err
        try:
            InlineTrans().apply(call)
        # Every failure to inline is a refusal, whatever its class: a name
        # the frontend read as a call may resolve to a datum, which PSyclone
        # reports with a TypeError, and an expression its machinery cannot
        # handle surfaces as whatever that machinery raises.
        except Exception as err:  # pylint: disable=broad-except
            first = (f"; bringing it into the container was refused "
                     f"first: {refusal}") if refusal else ""
            kind = ("" if isinstance(err, TransformationError)
                    else f"{type(err).__name__}: ")
            raise TransformationError(
                f"LFRicKokkosTrans cannot inline the call to '{name}' in "
                f"'{schedule.name}': {kind}{err}{first}") from err


__all__ = ["LFRicKokkosBoundMixin"]
