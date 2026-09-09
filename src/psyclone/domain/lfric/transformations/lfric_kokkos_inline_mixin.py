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

"""Bring a called subroutine's body into the kernel before capturing it.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

The generated region is a C++ function; there is no Fortran for it to call
into, so a kernel that calls one must have the callee's statements made its
own before anything else looks at the body. That rewrite is what this mixin
performs, and the refusal when it cannot is what it raises. It is placed
first among the rewrites for a reason: an inlined callee brings its own
loops, its own locals and its own array sections with it, and every later
rule -- sections, bounds, locals, constants -- has to be asked about the body
that results rather than the one the file holds.

The inlining itself is
:py:class:`~psyclone.psyir.transformations.InlineTrans`, not a rewrite of our
own. Everything it refuses is refused here with its own words kept, because
those words name the thing to fix -- a container the callee's data lives in, a
module that was never read, an argument whose type does not match the formal
-- and a paraphrase would name less.
"""

from psyclone.domain.common.transformations import KernelModuleInlineTrans
from psyclone.psyir.nodes import (
    Call, Container, IntrinsicCall, Routine, ScopingNode)
from psyclone.psyir.symbols import (
    RoutineSymbol, Symbol, UnsupportedFortranType)
from psyclone.psyir.transformations import InlineTrans, TransformationError


class LFRicKokkosInlineMixin:
    """Inline every call a kernel body makes, or refuse to capture it.

    **Repeated to a fixed point, not run once.** Inlining a call can expose
    another: a helper that calls a second helper leaves that second call
    behind in the body it contributes. So calls are inlined one at a time and
    the body re-examined after each, until none is left.

    **Bounded, because nothing else bounds it.**
    :py:class:`~psyclone.psyir.transformations.InlineTrans` has no recursion
    check: inlining a routine that calls itself substitutes its body and
    leaves a call to it in the substituted copy, so a fixed point is never
    reached and the rewrite would grow the body until it exhausted the
    interpreter's stack or its memory. :py:attr:`_INLINE_LIMIT` is the whole
    of what turns that into a refusal naming the routine still to be inlined.

    **What is in scope.** A procedure of the kernel's own module is, since
    its body sits in the Container the call is made from. A procedure of a
    module the kernel names in a ``use`` is too, provided PSyclone can read
    that module's source: it is first brought into the kernel's Container by
    :py:class:`~psyclone.domain.common.transformations.KernelModuleInlineTrans`
    and then inlined like any other, and what it needs comes with it: a named
    constant its own module reads from a third travels into the kernel's
    Container beside the body. Anything else -- a callee whose module is not
    on the search path, so that PSyclone has only a name for it -- is out of
    scope and refused.

    Where a callee could not be brought in, the refusal says so as well as
    saying why the call could not be inlined. ``InlineTrans`` alone reports
    the symptom -- the body is in another Container -- and
    :py:meth:`_module_inline`'s reason is the one that names what to fix.

    Being in scope is not being inlinable, and the difference is PSyclone's
    to state rather than ours. A callee reading data private to its own
    module, one whose declarations depend on an argument the call site writes
    to first, one whose actual and formal types do not match: each is refused
    by ``InlineTrans`` in its own words, and this mixin adds only which call
    it was.

    **Not every Call is a call.** ``weights(index)`` is a function reference
    or an element of an array, and where the kernel's own file does not say
    which -- the name comes from a ``use`` whose module the frontend did not
    read -- the frontend leaves a
    :py:class:`~psyclone.psyir.nodes.Call` behind. Asking PSyclone to resolve
    such a callee can reach a datum instead of a routine and raise
    :py:exc:`TypeError` rather than a
    :py:class:`~psyclone.psyir.transformations.TransformationError`. That is
    caught and refused with the rest: a kernel this transformation cannot
    capture must be declined, not turned into a traceback out of a
    transformation the caller merely asked to validate.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The most calls this will inline into one kernel body. A budget rather
    #: than a nesting depth, because the calls are inlined one at a time: a
    #: chain of helpers spends one of these per link, and a routine that calls
    #: itself spends them all and is then refused by name. Eight is chosen to
    #: be past any hand-written LFRic kernel's call chain and far short of
    #: what would take a runaway rewrite out of memory.
    _INLINE_LIMIT = 8

    #: The attributes a formal may carry and still be given its partial
    #: datatype by :py:meth:`_relax_target_arguments`. ``TARGET`` is the one
    #: that makes the declaration unmodelled in the first place; the other
    #: two are matched because they are already in the partial datatype --
    #: ``DIMENSION`` as its shape and ``INTENT`` as the symbol's argument
    #: interface -- and so are not attributes being dropped. Matched by
    #: prefix, since both are written with a parenthesised value.
    _BINDABLE_ATTRIBUTES = ("TARGET", "DIMENSION", "INTENT")

    @staticmethod
    def _callee_name(call):
        """Name the routine ``call`` calls, for a message.

        :param call: the call to name the callee of.
        :type call: :py:class:`psyclone.psyir.nodes.Call`

        :returns: the callee's name, or a stand-in where the call carries no
            routine symbol at all.
        :rtype: str
        """
        routine = call.routine
        return routine.symbol.name if routine else "an unnamed routine"

    @classmethod
    def _pending_calls(cls, schedule):
        """List the calls in ``schedule`` that still have to be inlined.

        An :py:class:`~psyclone.psyir.nodes.IntrinsicCall` is not one of them:
        the backend writes an intrinsic itself, so it is a call in PSyIR's
        terms and not in the generated region's.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: the non-intrinsic calls the body makes, in tree order.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Call`]
        """
        return [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]

    @classmethod
    def _inline_calls(cls, schedule):
        """Replace every call in ``schedule`` with the callee's statements.

        The body is re-walked after each inlining rather than the calls being
        collected once, because inlining rewrites the tree the collection was
        taken from: a call the callee itself makes did not exist when the walk
        ran, and a call sharing a statement with the one just inlined may have
        been moved out of it.

        Each callee is prepared before it is inlined: brought into the
        Container the call is made from by :py:meth:`_module_inline`, and
        then relaxed by :py:meth:`_relax_target_arguments`, which gives a
        formal declared ``TARGET`` the type the frontend parsed out of that
        declaration.

        :param schedule: the kernel schedule to rewrite in place.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :raises TransformationError: if a call cannot be inlined, in which
            case the reason is the one PSyclone gives for it, followed by the
            reason the callee was not brought into the Container first where
            there is one. A :py:exc:`TypeError` from resolving the callee is
            one of those reasons: a name the frontend read as a call may
            resolve to a datum, which PSyclone reports by failing to
            specialise the symbol rather than by refusing the
            transformation.
        :raises TransformationError: if calls are still left after
            :py:attr:`_INLINE_LIMIT` of them have been inlined, which is what
            a self-recursive callee looks like from here.
        """
        for _ in range(cls._INLINE_LIMIT):
            pending = cls._pending_calls(schedule)
            if not pending:
                return
            call = pending[0]
            name = cls._callee_name(call)
            refusal = cls._module_inline(call)
            cls._relax_target_arguments(call)
            try:
                InlineTrans().apply(call)
            except (TransformationError, TypeError) as err:
                first = (f"; bringing it into the container was refused "
                         f"first: {refusal}") if refusal else ""
                raise TransformationError(
                    f"LFRicKokkosTrans cannot inline the call to '{name}' in "
                    f"'{schedule.name}': {err}{first}") from err
        names = ", ".join(sorted(
            {cls._callee_name(call)
             for call in cls._pending_calls(schedule)}))
        raise TransformationError(
            f"LFRicKokkosTrans inlined {cls._INLINE_LIMIT} calls into "
            f"'{schedule.name}' and found the call to {names} still there: "
            f"a routine that calls itself is never inlined away, and no "
            f"kernel this is meant for calls that deeply.")

    @classmethod
    def _module_inline(cls, call):
        """Bring ``call``'s callee into the Container the call is made from.

        A callee reached through a ``use`` has its body in another Container,
        which is by itself a refusal from
        :py:class:`~psyclone.psyir.transformations.InlineTrans`. Moving it
        first is what puts a procedure of a ``use``d module in scope, and it
        carries the callee's own imports and declarations with it rather than
        leaving them behind -- a named constant the callee reads from a third
        module included, which is why a two-container callee such as LFRic's
        ``face_from_face_selector`` is reachable at all.

        **The callee's symbol is specialised first.** ``selector(face)`` in an
        expression is a function reference or an element of an array, and
        where the kernel's own file does not settle which the frontend leaves
        the name an unspecialised :py:class:`~psyclone.psyir.symbols.Symbol`.
        :py:class:`~psyclone.domain.common.transformations.KernelModuleInlineTrans`
        reads such a symbol at the call site as a datum of the callee's name
        and refuses to shadow it, which is a refusal about what the frontend
        could tell rather than about the code. A ``Call``'s callee is a
        routine whether or not the frontend could say so, so the symbol is
        made one. Only a bare ``Symbol`` is: a
        :py:class:`~psyclone.psyir.symbols.DataSymbol` is a name PSyclone has
        typed as data, and specialising that would assert something the
        source does not support.

        It is attempted rather than required, and a refusal is returned rather
        than raised: a callee that is already local is refused for being
        "already module inlined", which is the successful case, and where the
        callee did reach the Container the message worth having is the one
        :py:meth:`_inline_calls` then gets from ``InlineTrans`` about the call
        itself. So the refusal is returned only when the Container does not
        hold the callee afterwards, which is the case where it explains
        something ``InlineTrans``'s message alone does not. A
        :py:exc:`TypeError` from resolving the callee is returned on the same
        terms.

        :param call: the call whose callee is to be brought in.
        :type call: :py:class:`psyclone.psyir.nodes.Call`

        :returns: why the callee was not brought into the call's Container, or
            ``None`` where it is in it.
        :rtype: Optional[str]
        """
        routine = call.routine
        # pylint: disable-next=unidiomatic-typecheck
        if routine is not None and type(routine.symbol) is Symbol:
            routine.symbol.specialise(RoutineSymbol)
        try:
            KernelModuleInlineTrans().apply(call)
            return None
        except (TransformationError, TypeError) as err:
            return None if cls._callee_is_local(call) else str(err)

    @classmethod
    def _callee_is_local(cls, call):
        """Say whether ``call``'s Container holds a routine of that name.

        Asked after :py:meth:`_module_inline` has tried, this separates the
        callee that was already in the Container -- refused for being "already
        module inlined", which is success -- from the one that could not be
        moved into it. It is asked of the tree rather than of the refusal's
        wording, which is another transformation's to change.

        :param call: the call to look for the callee of.
        :type call: :py:class:`psyclone.psyir.nodes.Call`

        :returns: whether the Container the call is made from holds a routine
            named as the callee is.
        :rtype: bool
        """
        return bool(cls._local_callees(call))

    @classmethod
    def _local_callees(cls, call):
        """Return the routines of ``call``'s Container named as its callee.

        There is at most one where the callee is a plain procedure, and more
        than one where the name is a generic interface: which specific the
        call resolves to is PSyclone's to decide, so both are returned and
        the caller says what it does with them.

        :param call: the call to look for the callee of.
        :type call: :py:class:`psyclone.psyir.nodes.Call`

        :returns: the routines the Container the call is made from holds
            under the callee's name.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Routine`]
        """
        container = call.ancestor(Container)
        if container is None:
            return []
        name = cls._callee_name(call).lower()
        return [routine
                for routine in container.walk(Routine, stop_type=Routine)
                if routine.name.lower() == name]

    @staticmethod
    def _declaration_attributes(declaration):
        """List the attributes of a Fortran declaration, less its type.

        The text is the one
        :py:class:`~psyclone.psyir.symbols.UnsupportedFortranType` kept of a
        declaration PSyclone could not model, so it is read rather than
        re-parsed. Only the specification part -- what stands before ``::``
        -- carries attributes; the entity part after it carries the name and
        any shape given with it. Commas inside parentheses do not separate
        attributes, which is why the split is made at depth zero:
        ``DIMENSION(n, 2)`` is one attribute and not two.

        Each attribute is returned upper-cased and stripped of every space,
        so that ``intent ( in )`` and ``INTENT(IN)`` are one string.

        A declaration written without ``::`` states no attributes, and
        needs no case of its own: the split leaves the whole declaration as
        one field, and dropping the type specification drops it.

        :param str declaration: the declaration text to read.

        :returns: the declaration's attributes, the leading type
            specification dropped, or an empty list where it declares no
            attributes at all.
        :rtype: List[str]
        """
        attributes = []
        current = ""
        depth = 0
        for character in declaration.split("::")[0]:
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            if character == "," and depth == 0:
                attributes.append(current)
                current = ""
            else:
                current += character
        attributes.append(current)
        return ["".join(attribute.split()).upper()
                for attribute in attributes[1:]]

    @classmethod
    def _relax_target_arguments(cls, call):
        """Give every ``TARGET`` formal of ``call``'s callee its own type.

        ``TARGET`` says that a pointer somewhere may be aimed at the actual.
        Inlining the callee's statements into the caller neither creates
        such a pointer nor invalidates one that exists, so the attribute has
        nothing to say about whether the body may be substituted -- but the
        PSyIR does not model it, so the formal arrives as an
        :py:class:`~psyclone.psyir.symbols.UnsupportedFortranType` and
        :py:class:`~psyclone.psyir.transformations.InlineTrans` refuses the
        routine for having an argument of a type it does not know. Replacing
        such a formal's type with the partial datatype the frontend did
        parse is what removes that refusal, and it removes only that one:
        ``permit_unsupported_type_args`` is not passed, so a formal carrying
        any other unmodelled attribute -- ``POINTER``, ``ALLOCATABLE``,
        ``OPTIONAL``, ``VALUE`` -- is left as it is and refused as before.
        This is what makes the vertical-support helpers of LFRic's FFSL
        schemes, whose read column is declared ``TARGET``, inlinable at all.

        The rewrite is made on the callee as the call site sees it, after
        :py:meth:`_module_inline` has run: a callee reached through a ``use``
        is by then a copy in the caller's Container, and a callee of the
        kernel's own module is in the copy of the file
        :py:meth:`_rooted_copy` took. Neither is the tree the frontend
        parsed, so a later capture of another kernel calling the same helper
        meets the routine as its own module declares it.

        The partial datatype is taken rather than copied: it is part of the
        declaration being replaced, which nothing holds afterwards.

        :param call: the call whose callee is to be relaxed.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        """
        for routine in cls._local_callees(call):
            for scope in routine.walk(ScopingNode):
                for symbol in scope.symbol_table.symbols:
                    datatype = symbol.datatype if symbol.is_argument else None
                    if not isinstance(datatype, UnsupportedFortranType):
                        continue
                    attributes = cls._declaration_attributes(
                        datatype.declaration)
                    # Two questions with one answer: a declaration the
                    # frontend could parse nothing of leaves nothing to put
                    # in the symbol's place, and one that is unmodelled for
                    # some other reason is not this rewrite's to relax.
                    if (datatype.partial_datatype is None or
                            "TARGET" not in attributes):
                        continue
                    if all(attribute.startswith(cls._BINDABLE_ATTRIBUTES)
                           for attribute in attributes):
                        symbol.datatype = datatype.partial_datatype

    @staticmethod
    def _rooted_copy(schedule):
        """Return a copy of ``schedule`` that keeps its scope chain.

        The copy is of the whole file the kernel was read from, not of the
        schedule alone, and the routine is then found again by its position.
        A Routine copied on its own is detached from the FileContainer and
        the Container above it, so every name the kernel module ``use``d at
        module level -- an ``integer, parameter`` an array subscript reads,
        say -- is no longer resolvable from the copy. A rule asked about the
        copy would then be answering about a scope chain the kernel does not
        have, and the answer is not always a refusal: looking such a name up
        raises where nothing catches it.

        :param schedule: the kernel schedule to copy.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: the copied schedule, inside a copy of its own file.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`

        """
        root = schedule.root
        index = root.walk(Routine).index(schedule)
        return root.copy().walk(Routine)[index]

    @classmethod
    def _inlined_copy(cls, schedule):
        """Return a copy of ``schedule`` with its calls inlined.

        The copy is of the whole file the kernel was read from, not of the
        schedule alone. Inlining looks for the callee's implementation in the
        call site's ancestor Container, so a schedule copied on its own would
        be refused for having no Container to look in -- a refusal about the
        copy rather than about the kernel, and a wrong answer.

        :param schedule: the kernel schedule to predict the rewrite for.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: the copied schedule, with every call inlined.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`

        :raises TransformationError: if a call cannot be inlined, by
            :py:meth:`_inline_calls`.
        """
        copy = cls._rooted_copy(schedule)
        cls._inline_calls(copy)
        return copy


__all__ = ["LFRicKokkosInlineMixin"]
