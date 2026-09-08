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

"""Write the intrinsics a Kokkos region spells differently from C.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.psyir.backend.kokkos.KokkosWriter` ahead of
:py:class:`~psyclone.psyir.backend.c.CWriter` rather than instantiated. It
holds no state of its own: every method either falls through to the C writer's
handler by ``super()`` or answers from the tables below, and the one piece of
region knowledge it needs -- the C type a Fortran kind maps to -- it reaches
through ``self._kind_c_type``, which the writer provides.

It is separate from the writer for the reason ``kokkos_launch`` is: the
intrinsics are the part of the back-end that grows one entry at a time as
kernels are captured, while the writer's own job does not.
"""

from psyclone.psyir.backend.c_intrinsics_mixin import _is_real_argument
from psyclone.psyir.backend.kokkos_array_intrinsics import (
    KokkosArrayIntrinsics)
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import IntrinsicCall, Reference
from psyclone.psyir.symbols import DataSymbol


class KokkosIntrinsicsMixin:
    """Spell Fortran's intrinsics as Kokkos does, at the kind that was asked.

    The C writer is right about the shape of every intrinsic it knows and
    wrong about two things a Kokkos region cares about: it cannot see the
    width a Fortran kind asks for, and it emits unqualified names that are
    host functions. Everything here is one of those two corrections, or an
    intrinsic the C writer has no handler for at all.
    """

    #: Intrinsics that become a plain ``Kokkos::`` function call, by the name
    #: Kokkos gives them. Qualification is required for device code -- an
    #: unqualified ``sqrt`` is a host function -- and is correct on the host
    #: too. ``ABS`` and ``MOD`` are absent because both are spelt differently
    #: for an integer argument; they are dispatched in
    #: :py:meth:`intrinsiccall_node` instead.
    _KOKKOS_FUNCTIONS = {
        IntrinsicCall.Intrinsic.ACOS: "acos",
        IntrinsicCall.Intrinsic.ASIN: "asin",
        IntrinsicCall.Intrinsic.ATAN: "atan",
        IntrinsicCall.Intrinsic.ATAN2: "atan2",
        IntrinsicCall.Intrinsic.COS: "cos",
        IntrinsicCall.Intrinsic.EXP: "exp",
        IntrinsicCall.Intrinsic.LOG: "log",
        IntrinsicCall.Intrinsic.SIGN: "copysign",
        IntrinsicCall.Intrinsic.SIN: "sin",
        IntrinsicCall.Intrinsic.SQRT: "sqrt",
        IntrinsicCall.Intrinsic.TAN: "tan",
        }

    #: Intrinsics that return a Fortran integer from a floating-point
    #: function, so that the call needs an ``(int)`` cast round it. Fortran's
    #: ``NINT`` rounds a half away from zero, which is ``round`` and not
    #: ``nearbyint``: the latter rounds a half to even under the default
    #: rounding mode, so ``NINT(2.5)`` would give 2 rather than 3.
    _KOKKOS_CAST_FUNCTIONS = {
        IntrinsicCall.Intrinsic.FLOOR: "floor",
        IntrinsicCall.Intrinsic.NINT: "round",
        }

    #: Intrinsics written as a cast, and the C types a resolved kind is
    #: allowed to give for each. The check is not redundant: PSyIR infers the
    #: precision of a kindless ``real(i)`` from its *argument*, reporting
    #: ``Scalar<REAL, Reference['i_def']>``, so resolving that kind would cast
    #: to ``int`` and lose the value. A kind whose C type does not belong to
    #: the intrinsic being cast to is therefore treated as unresolved, and the
    #: C writer's kind-blind answer -- the widest of the intrinsic -- stands.
    _KOKKOS_CAST_TYPES = {
        IntrinsicCall.Intrinsic.REAL: ("double", "float"),
        IntrinsicCall.Intrinsic.INT: ("int",),
        }

    #: Variadic intrinsics folded pairwise into nested binary calls. Unlike
    #: ``fmax``, ``Kokkos::max`` is type-generic, so the integer case
    #: :py:class:`~psyclone.psyir.backend.c.CWriter` refuses is generated
    #: here.
    _KOKKOS_FOLDS = {
        IntrinsicCall.Intrinsic.MAX: "max",
        IntrinsicCall.Intrinsic.MIN: "min",
        }

    #: Intrinsics that ask about an array's declared shape rather than about
    #: its values. Lowering answers all three from the region's own View
    #: descriptions and none of them survives to be written, so they are not
    #: probed: asking the writer would report a refusal no generated region
    #: can reach.
    _QUERIES = (IntrinsicCall.Intrinsic.LBOUND,
                IntrinsicCall.Intrinsic.UBOUND,
                IntrinsicCall.Intrinsic.SIZE)

    #: The name every argument of a probed intrinsic is given. It is never
    #: generated into a region: the probe's answer is discarded and only
    #: whether it raised is kept.
    _PROBE = "_kae_probe"

    def literal_node(self, node) -> str:
        """Write a literal at the width its own Fortran kind has.

        ``2.0_r_solver`` is a ``float`` in a single-precision build, and C++
        would otherwise read the generated ``2.0`` as a ``double`` and promote
        the whole expression around it.

        :param node: the literal to write.
        :type node: :py:class:`psyclone.psyir.nodes.Literal`

        :returns: the C representation of the literal.
        :rtype: str
        """
        text = super().literal_node(node)
        if self._kind_c_type(node.datatype) == "float":
            return f"{text}f"
        return text

    def intrinsiccall_node(self, node) -> str:
        """Write an intrinsic call as Kokkos spells it.

        The C writer is right about the shape of every intrinsic it knows and
        wrong about two things a Kokkos region cares about: it cannot see the
        width a Fortran kind asks for, and it emits unqualified names that are
        host functions. Each case this method does not recognise falls back to
        it, so the set of intrinsics supported here is the C writer's plus
        ``EPSILON``, plus integer ``MAX`` and ``MIN``.

        :param node: the intrinsic call to write.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the C++/Kokkos representation of the call.
        :rtype: str

        :raises VisitorError: if the intrinsic needs a kind the region did
            not describe, or if the C writer has no handler for it.
        """
        for handler in (self._kokkos_cast, self._kokkos_numeric_limit,
                        self._kokkos_function):
            written = handler(node)
            if written is not None:
                return written
        return super().intrinsiccall_node(node)

    def _kokkos_cast(self, node):
        """Write ``REAL`` or ``INT`` at the width its own kind asks for.

        ``real(x, r_solver)`` is a ``float`` in a single-precision build, and
        the C writer casts every real to ``double``. A kind the region did not
        describe is left to it: the probe a caller uses to ask which
        intrinsics are supported replaces every argument with a bare literal,
        so an unresolved kind is that probe's normal case rather than an
        error. So is a kind that resolves to the wrong intrinsic, for the
        reason given at :py:attr:`_KOKKOS_CAST_TYPES`.

        :param node: the intrinsic call to write.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the cast, or ``None`` if this is not a cast, if its kind is
            unresolved, or if its arity is one the C writer should refuse.
        :rtype: Optional[str]
        """
        allowed = self._KOKKOS_CAST_TYPES.get(node.intrinsic)
        if allowed is None or len(node.arguments) not in (1, 2):
            return None
        c_type = self._kind_c_type(node.datatype)
        if c_type not in allowed:
            return None
        return f"({c_type}){self._visit(node.arguments[0])}"

    def _kokkos_numeric_limit(self, node):
        """Write ``EPSILON`` as the Kokkos numeric trait for its kind.

        The argument is consumed for its type and never visited, since the
        intrinsic asks a question about a type rather than about a value.

        :param node: the intrinsic call to write.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the trait, or ``None`` if this is not ``EPSILON``.
        :rtype: Optional[str]

        :raises VisitorError: if the region described no kind for the
            argument, leaving the trait with no type to instantiate.
        """
        if node.intrinsic is not IntrinsicCall.Intrinsic.EPSILON:
            return None
        c_type = self._kind_c_type(node.arguments[0].datatype)
        if c_type is None:
            raise VisitorError(
                "EPSILON needs the width of its argument's kind, which this "
                "region does not describe. Add the kind to the region's "
                "'kind_types'.")
        return f"Kokkos::Experimental::epsilon_v<{c_type}>"

    def _kokkos_function(self, node):
        """Write an intrinsic that becomes a qualified Kokkos function call.

        :param node: the intrinsic call to write.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the call, or ``None`` if Kokkos has no function for this
            intrinsic and the C writer's own answer should stand.
        :rtype: Optional[str]

        :raises VisitorError: if a folded intrinsic has fewer than two
            arguments, which no valid Fortran ``MAX`` or ``MIN`` has.
        """
        intrinsic = node.intrinsic
        if intrinsic in self._KOKKOS_FOLDS:
            if len(node.arguments) < 2:
                raise VisitorError(
                    f"The Kokkos back-end can only fold "
                    f"'{intrinsic.name}' over 2 or more arguments, but found "
                    f"{len(node.arguments)}.")
            return self._fold(self._KOKKOS_FOLDS[intrinsic], node)
        if intrinsic in self._KOKKOS_CAST_FUNCTIONS and \
                len(node.arguments) in (1, 2):
            return self._kokkos_cast_function(node)
        name = self._function_name(node)
        if name is None:
            return None
        arguments = ", ".join(self._visit(argument)
                              for argument in node.arguments)
        return f"Kokkos::{name}({arguments})"

    def _kokkos_cast_function(self, node):
        """Write a rounding intrinsic as a cast round a Kokkos function.

        Fortran's ``NINT`` and ``FLOOR`` take an optional second argument
        naming the integer kind of their result. It is not generated -- the
        cast is written as ``(int)`` and nothing else -- so it is checked
        instead: a kind the region maps to some other width would be
        discarded silently, and a kernel asking for a 64-bit result would get
        a 32-bit one that compiles and truncates.

        The kind is read from the argument rather than from the call's type.
        PSyIR gives ``NINT(x, i_def)`` an undefined precision, the result kind
        of these intrinsics not being carried into the type they report, so
        the second argument's own symbol is what names the width asked for.

        :param node: the intrinsic call to write.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the cast call.
        :rtype: str

        :raises VisitorError: if the call names a result kind the region maps
            to a C type other than the one the cast is written with.
        """
        name = self._KOKKOS_CAST_FUNCTIONS[node.intrinsic]
        allowed = self._KOKKOS_CAST_TYPES[IntrinsicCall.Intrinsic.INT]
        c_type = None
        if len(node.arguments) == 2 and isinstance(node.arguments[1],
                                                   Reference):
            c_type = self._kind_types.get(node.arguments[1].symbol.name)
        if c_type is not None and c_type not in allowed:
            raise VisitorError(
                f"'{node.intrinsic.name}' is written as a cast to "
                f"'{allowed[0]}', but this region maps the result kind it was "
                f"given to '{c_type}'. Generating the cast would truncate the "
                "value the kernel asked for.")
        argument = self._visit(node.arguments[0])
        return f"({allowed[0]})Kokkos::{name}({argument})"

    def unsupported_intrinsics(self, schedule, kind_types=()):
        """Return the intrinsics in a body that this writer cannot spell.

        A transformation has to refuse a body the writer will refuse, and the
        only thing that knows what the writer can spell is the writer: the
        tables above are searched by several handlers, one of which builds its
        map as a local, so a caller cannot read them and a second list kept
        beside them would be a second list to keep in step. Each call is asked
        of the writer instead, and what it refuses is reported by name.

        The call is asked as a *probe*: every argument is replaced by a
        reference of that argument's own type, so that an intrinsic answered
        from its argument's kind -- ``EPSILON`` -- is still answered, while an
        array the region never described is not looked up and an argument that
        is itself unwritable does not stand in for the call. The intrinsics of
        the array-valued tier are not asked at all: they are not written by a
        handler, and lowering replaces them before the writer meets one.

        :param schedule: the body to search.
        :type schedule: :py:class:`psyclone.psyir.nodes.Node`
        :param kind_types: the ``(kind name, C type)`` pairs the region will
            be generated with.
        :type kind_types: Iterable[Tuple[str, str]]

        :returns: one ``NAME/arity`` entry per intrinsic refused, in the order
            first met and without repetition.
        :rtype: Tuple[str, ...]
        """
        self._kind_types = dict(kind_types)
        self._views = {}
        refused = []
        for call in schedule.walk(IntrinsicCall):
            if call.intrinsic in self._QUERIES \
                    or KokkosArrayIntrinsics.handles(call):
                continue
            probe = call.copy()
            for argument in probe.arguments:
                argument.replace_with(Reference(
                    DataSymbol(self._PROBE, argument.datatype)))
            try:
                self.intrinsiccall_node(probe)
            except (VisitorError, ValueError, KeyError, NotImplementedError):
                refused.append(
                    f"{call.intrinsic.name}/{len(call.arguments)}")
        return tuple(dict.fromkeys(refused))

    @classmethod
    def _function_name(cls, node):
        """Return the Kokkos function an intrinsic call is written with.

        ``ABS`` and ``MOD`` are spelt by their argument's type, exactly as in
        the C writer: ``Kokkos::abs`` truncates a real and ``Kokkos::fabs``
        returns a double for an integer. Integer ``MOD`` has no function at
        all -- it is the ``%`` operator, which needs no qualification -- so it
        is returned to the C writer.

        :param node: the intrinsic call to write.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the unqualified function name, or ``None`` if this
            intrinsic is not written as a Kokkos function call.
        :rtype: Optional[str]
        """
        if node.intrinsic is IntrinsicCall.Intrinsic.ABS:
            return "fabs" if _is_real_argument(node.arguments[0]) else "abs"
        if node.intrinsic is IntrinsicCall.Intrinsic.MOD:
            return "fmod" if _is_real_argument(node.arguments[0]) else None
        return cls._KOKKOS_FUNCTIONS.get(node.intrinsic)

    def _fold(self, name, node):
        """Fold a variadic intrinsic into nested binary Kokkos calls.

        Folded right to left, so that three arguments give
        ``Kokkos::max(a, Kokkos::max(b, c))``.

        :param str name: the unqualified Kokkos function to fold with.
        :param node: the intrinsic call to write.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: the nested calls.
        :rtype: str
        """
        operands = [self._visit(argument) for argument in node.arguments]
        folded = operands[-1]
        for operand in reversed(operands[:-1]):
            folded = f"Kokkos::{name}({operand}, {folded})"
        return folded


__all__ = ["KokkosIntrinsicsMixin"]
