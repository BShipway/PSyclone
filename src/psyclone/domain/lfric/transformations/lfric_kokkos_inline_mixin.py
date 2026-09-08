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
from psyclone.psyir.nodes import Call, IntrinsicCall, Routine
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
    and then inlined like any other. Anything else -- a callee whose module is
    not on the search path, so that PSyclone has only a name for it -- is out
    of scope and refused.

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
            case the reason is the one PSyclone gives for it. A
            :py:exc:`TypeError` from resolving the callee is one of those
            reasons: a name the frontend read as a call may resolve to a
            datum, which PSyclone reports by failing to specialise the symbol
            rather than by refusing the transformation.
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
            cls._module_inline(call)
            try:
                InlineTrans().apply(call)
            except (TransformationError, TypeError) as err:
                raise TransformationError(
                    f"LFRicKokkosTrans cannot inline the call to '{name}' in "
                    f"'{schedule.name}': {err}") from err
        names = ", ".join(sorted(
            {cls._callee_name(call)
             for call in cls._pending_calls(schedule)}))
        raise TransformationError(
            f"LFRicKokkosTrans inlined {cls._INLINE_LIMIT} calls into "
            f"'{schedule.name}' and found the call to {names} still there: "
            f"a routine that calls itself is never inlined away, and no "
            f"kernel this is meant for calls that deeply.")

    @staticmethod
    def _module_inline(call):
        """Bring ``call``'s callee into the Container the call is made from.

        A callee reached through a ``use`` has its body in another Container,
        which is by itself a refusal from
        :py:class:`~psyclone.psyir.transformations.InlineTrans`. Moving it
        first is what puts a procedure of a ``use``d module in scope, and it
        carries the callee's own imports and declarations with it rather than
        leaving them behind.

        It is attempted rather than required, and its refusals are dropped
        rather than reported: a callee that is already local is refused for
        being "already module inlined", which is the successful case, and for
        any other refusal the message worth having is the one
        :py:meth:`_inline_calls` then gets from ``InlineTrans`` about the
        call itself. Reporting this one instead would name a rewrite the
        reader never asked for. A :py:exc:`TypeError` from resolving the
        callee is dropped for the same reason and on the same terms: it is
        raised again where the call is inlined, and refused there.

        :param call: the call whose callee is to be brought in.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        """
        try:
            KernelModuleInlineTrans().apply(call)
        except (TransformationError, TypeError):
            pass

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
