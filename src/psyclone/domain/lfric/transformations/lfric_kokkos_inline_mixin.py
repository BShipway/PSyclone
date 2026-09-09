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
    Call, Container, IntrinsicCall, Reference, Routine)
from psyclone.psyir.symbols import (
    ContainerSymbol, ImportInterface, RoutineSymbol, Symbol)
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
    Container beside the body. A procedure that procedure calls is in scope
    too, whether it is a sibling of its own module or a name that module
    reaches by a ``use`` of its own: a sibling is inlined where it already
    can be, in the Container the two share, before the body travels.
    Anything else -- a callee whose module is not on the search path, so that
    PSyclone has only a name for it -- is out of scope and refused.

    **Data of the callee's module is not in scope, and a procedure of it is
    not data.** A callee reading a variable its own module declares --
    LFRic's ``chi2xyz`` and ``chi2xyz_rot_mat`` -- is refused, and refused
    for the right reason: the region would need the PSy layer to import that
    name and pass it, which is what ``LFRicKokkosConstantsMixin`` does for a
    variable of the *kernel's* module and cannot do for one the callee's
    module keeps private.

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

        **A callee that calls its own module is made self-contained first.**
        ``KernelModuleInlineTrans`` refuses a routine that reads anything
        declared beside it in its own module, a sibling *procedure* included:
        LFRic's ``crosses_panel_edge`` calls ``rotated_panel_neighbour``,
        which calls ``panel_neighbour``, all three of
        ``panel_edge_support_mod``. Nothing about that chain is out of scope
        -- each link is a routine whose source PSyclone has read -- and the
        one place each call *is* already inlinable is the container they
        share. So :py:meth:`_absorb_own_module_calls` inlines them there,
        before the body travels, and what arrives in the kernel's Container
        is a routine with no sibling left to reach for.

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
        cls._absorb_own_module_calls(call)
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
    def _absorb_own_module_calls(cls, call):
        """Inline into ``call``'s callee the calls it makes to its own module.

        A routine reached through a ``use`` cannot be brought into the
        kernel's Container while it still names something declared beside it
        in its own module: ``KernelModuleInlineTrans`` would have to add an
        import at the call site for that name, and it does not. Where the
        name is a *procedure* of that module there is nothing to import,
        because the body is available and the call sites are all in the one
        Container -- so it is inlined there instead, where
        :py:class:`~psyclone.psyir.transformations.InlineTrans` needs no
        Container crossed and no symbol carried across one.

        The loop runs to a fixed point rather than once, because a sibling
        brings its own siblings with it: absorbing LFRic's
        ``rotated_panel_neighbour`` into ``crosses_panel_edge`` leaves the
        call to ``panel_neighbour`` behind in the statements it contributed,
        and that is a call to the same module again. It is bounded by
        :py:attr:`_INLINE_LIMIT` for the reason the outer loop is, and the
        callee is never absorbed into itself: a routine that calls itself is
        reported by name by :py:meth:`_inline_calls` rather than rewritten
        here.

        **The tree rewritten is the module's, and every step is rehearsed
        on a copy of it first.** ``KernelModuleInlineTrans`` reads the callee
        through :py:meth:`~psyclone.psyir.nodes.Call.get_callees`, whose
        answer is the Container the module was parsed into and is held for
        the call, so the preparation has to be made in that Container, in
        place, to be seen at all. But ``InlineTrans`` does not always refuse
        before it starts: LFRic's ``physics_mappings_alg_mod`` reaches an
        ``AttributeError`` inside it on an actual argument that is an
        expression, and a module left half-rewritten in that cache would be
        read as the file's own text by every later kernel. Each absorption is
        therefore made twice -- once on a copy of the Container, which is
        thrown away, and then, only if that worked, on the Container itself.
        What the cache holds is the file's routine or one whole absorption
        further on, never the middle of one.

        The rewrite is meaning-preserving and idempotent -- a call replaced by
        the statements it stood for, and no call left to replace a second time
        -- so a later kernel calling the same routine reads a body that says
        what the file said.

        Anything raised by the rehearsal ends the loop rather than being
        reported: the callee is left as it was, and the reason the kernel
        cannot have it is the one :py:meth:`_module_inline` then gets from
        ``KernelModuleInlineTrans`` for the whole body. That is why a sibling
        ``InlineTrans`` will not take -- one declaring a static local, say --
        is not a failure here but a call still in the body, which the refusal
        then names. The exception caught is any, not the refusal alone,
        because a preparation that is optional has no business turning a
        refusal into a crash.

        The Container asked for siblings is the nearest one, which is the
        module for a procedure of a module and the file for a procedure of
        neither. Both are answers to the same question: which routines the
        callee can reach without a name being carried anywhere.

        :param call: the call whose callee is to be made self-contained.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        """
        try:
            callees = call.get_callees()
        except Exception:                        # pylint: disable=W0703
            # A callee PSyclone cannot resolve has nothing to prepare, and
            # the caller reports the name it could not reach.
            return
        for callee in callees:
            container = callee.ancestor(Container)
            name = callee.name.lower()
            for _ in range(cls._INLINE_LIMIT):
                position = cls._own_module_call(container, name)
                if position is None:
                    break
                sibling = cls._callee_name(
                    container.walk(Call)[position]).lower()
                trial = container.copy()
                try:
                    cls._localise_imports(cls._sibling(trial, sibling))
                    InlineTrans().apply(trial.walk(Call)[position])
                    cls._localise_imports(cls._sibling(container, sibling))
                    InlineTrans().apply(container.walk(Call)[position])
                except Exception:                # pylint: disable=W0703
                    break

    @classmethod
    def _own_module_call(cls, container, name):
        """Find the first call the routine ``name`` makes to a sibling.

        The answer is a position in ``container.walk(Call)`` rather than the
        call itself, because the caller works on a copy of the Container and
        a copy holds no node the original does. Walk order is structural, so
        the same position in the copy is the same call.

        :param container: the Container the callee belongs to.
        :type container: :py:class:`psyclone.psyir.nodes.Container`
        :param str name: the callee's name, lowercased.

        :returns: where in the Container's calls the first call to a sibling
            of the callee is, or ``None`` if the callee makes none.
        :rtype: Optional[int]
        """
        routines = container.walk(Routine, stop_type=Routine)
        callee = cls._sibling(container, name)
        siblings = {routine.name.lower() for routine in routines
                    if routine is not callee}
        calls = container.walk(Call)
        for inner in cls._pending_calls(callee):
            if cls._callee_name(inner).lower() not in siblings:
                continue
            for position, node in enumerate(calls):
                if node is inner:
                    return position
        return None

    @staticmethod
    def _sibling(container, name):
        """Name a routine of ``container``'s own scope.

        The Container asked always holds the routine asked for: it is the
        callee's own nearest Container, or the copy of it the caller is
        rewriting, and the sibling's name was read out of it a moment before.

        :param container: the Container to look in.
        :type container: :py:class:`psyclone.psyir.nodes.Container`
        :param str name: the routine's name, lowercased.

        :returns: the routine of that name.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return next((routine for routine
                     in container.walk(Routine, stop_type=Routine)
                     if routine.name.lower() == name), None)

    @classmethod
    def _localise_imports(cls, routine):
        """Move into ``routine``'s own table the imports it reads from above.

        :py:class:`~psyclone.psyir.transformations.InlineTrans` copies the
        routine it is inlining, and a
        :py:class:`~psyclone.psyir.nodes.Routine` copied on its own is
        detached from the Container whose ``use`` statements named half of
        what it reads. Every such name arrives at the call site *unresolved*,
        and the second call to the same routine is then refused for a clash
        between two unresolved symbols of that name -- a refusal about the
        copy rather than about the code, and one that would let a routine be
        absorbed once and not twice.

        Giving the routine its own import of each name it reads is what the
        copy then carries: ``panel_neighbour``, which reads ``W``, ``S``,
        ``E`` and ``N`` from its module's ``use reference_element_mod``, is
        rewritten to read them from a ``use`` of its own. That is the same
        Fortran, and the symbol at the call site is an import that a second
        import of the same name agrees with.

        A copy of the symbol is added rather than the symbol itself: the
        Container's own table is the one every other routine of the module
        reads that name through, and taking it out of that table would be a
        rewrite of routines this has no business in.

        :param routine: the routine about to be inlined into its own module.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        """
        table = routine.symbol_table
        for signature in routine.reference_accesses().all_signatures:
            name = signature.var_name
            if name in table:
                continue
            symbol = table.lookup(name, otherwise=None)
            if symbol is None or not symbol.is_import:
                continue
            source = symbol.interface.container_symbol
            if source.name in table:
                local_source = table.lookup(source.name)
            else:
                local_source = ContainerSymbol(source.name)
                table.add(local_source)
            local = symbol.copy()
            local.interface = ImportInterface(
                local_source, orig_name=symbol.interface.orig_name)
            table.add(local)
            for reference in routine.walk(Reference):
                if reference.symbol is symbol:
                    reference.symbol = local

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
        container = call.ancestor(Container)
        if container is None:
            return False
        name = cls._callee_name(call).lower()
        return any(routine.name.lower() == name
                   for routine in container.walk(Routine, stop_type=Routine))

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
