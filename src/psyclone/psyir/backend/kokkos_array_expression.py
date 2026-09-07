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

"""What an array-valued expression lowers to in the generated Kokkos source.

An array expression -- a whole-array assignment such as ``pv_at_quad = ...``,
a section such as ``a(i:j)``, or an array constructor -- has no counterpart in
C++. It lowers to a *nest*: one generated ``for`` per dimension of the shape
it produces, whose body assigns one element, writing either into an array the
caller named or into a temporary this module allocates.

Two parts of that live here. :py:class:`KokkosArrayExpressionMixin` holds the
per-node translation the writer performs once the nest exists -- the loops,
the element assignments and the subscripts they carry -- and is mixed into
:py:class:`~psyclone.psyir.backend.kokkos.KokkosWriter`. It is written as a
mixin rather than as free functions because every one of its methods is a
visitor handler, reached by name from
:py:class:`~psyclone.psyir.backend.visitor.PSyIRVisitor` and falling through
to :py:class:`~psyclone.psyir.backend.c.CWriter` by ``super()``; a handler
that is not on the writer is not called at all.

Split out of :py:mod:`psyclone.psyir.backend.kokkos` rather than added to it,
which was within thirty lines of the size a module of this project may reach.
"""

from psyclone.psyir.backend.kokkos_constant import KokkosConstant
from psyclone.psyir.nodes import (
    ArrayConstructor, ArrayReference, Reference)
from psyclone.psyir.symbols import ArrayType


class KokkosArrayExpressionMixin:
    """The writer's handling of arrays in a captured body.

    Mixed into :py:class:`~psyclone.psyir.backend.kokkos.KokkosWriter` ahead
    of :py:class:`~psyclone.psyir.backend.c.CWriter`, so that these handlers
    are found first and reach the C writer's by ``super()``. The state they
    read -- ``_views``, ``_kind_types``, ``_parallel_loops`` and
    ``_parallel_depth`` -- is established by the writer for the duration of
    one region and is documented there.
    """

    #: How many of the region's chosen loops the visitor is currently
    #: inside, and how far the generated source is indented. Both are
    #: counters the writer owns and both start at zero, so the mixin declares
    #: that starting value rather than annotating a name it never sets: a
    #: counter read before its owner has established it is at zero either
    #: way, and pylint can only check an attribute the class states.
    _depth = 0
    _parallel_depth = 0

    def _kind_c_type(self, datatype):
        """Return the C type this region generates for a datatype's kind.

        :param datatype: the datatype whose kind is to be resolved.
        :type datatype: :py:class:`psyclone.psyir.symbols.DataType`

        :returns: the C type, or ``None`` if the datatype names no kind that
            the region described.
        :rtype: Optional[str]
        """
        if isinstance(datatype, ArrayType):
            datatype = datatype.elemental_type
        precision = getattr(datatype, "precision", None)
        if not isinstance(precision, Reference):
            return None
        return self._kind_types.get(precision.symbol.name)

    def gen_declaration(self, symbol) -> str:
        """Declare a symbol at the width its Fortran kind actually has.

        The C writer maps every real onto ``double``, which would promote a
        single-precision kernel's locals without saying so. A symbol whose
        kind the region did not describe is left to it: the counter
        :py:class:`~psyclone.psyir.transformations.ArrayAssignment2LoopsTrans`
        introduces for a lowered array section has no named kind, and must
        still be generated as ``int``.

        :param symbol: the symbol to declare.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: the C declaration, without indentation or punctuation.
        :rtype: str
        """
        c_type = self._kind_c_type(symbol.datatype)
        if c_type is None:
            return super().gen_declaration(symbol)
        pointer = "* restrict " if symbol.is_array else ""
        return f"{c_type} {pointer}{symbol.name}"

    def loop_node(self, node) -> str:
        """Spread one of the region's chosen loops over the team.

        A loop the region named in :py:attr:`KokkosRegion.parallel_loops`
        becomes a ``TeamVectorRange`` ``parallel_for``; every other loop,
        including one nested inside this body, is left to
        :py:class:`~psyclone.psyir.backend.c.CWriter` as a serial ``for`` that
        each member runs on its own. The Fortran bound is inclusive and the
        Kokkos range is half-open, which is the ``+ 1`` on the stop
        expression.

        A ``team_barrier`` follows unconditionally. A statement after the loop
        may read what the loop wrote, and working out whether one does is a
        second dependence analysis this writer does not perform; under
        ``Kokkos::AUTO`` on the OpenMP backend the team has one member and the
        barrier costs nothing measurable.

        The lambda's parameter shadows the region-scope declaration of the
        loop variable, which stays because the same variable may also drive a
        serial loop in the same body. Shadowing a local with a lambda
        parameter is legal C++, and the generated code is not compiled with
        ``-Wshadow``.

        :param node: the loop in the captured body.
        :type node: :py:class:`psyclone.psyir.nodes.Loop`

        :returns: the ``TeamVectorRange`` launch and its barrier, or the
            serial ``for`` the C writer would have produced.
        :rtype: str
        """
        if not any(loop is node for loop in self._parallel_loops):
            return super().loop_node(node)

        start = self._visit(node.start_expr)
        stop = self._visit(node.stop_expr)
        self._parallel_depth += 1
        self._depth += 1
        body = "".join(self._visit(child) for child in node.loop_body)
        self._depth -= 1
        self._parallel_depth -= 1
        return (
            f"{self._nindent}Kokkos::parallel_for("
            f"Kokkos::TeamVectorRange(team, {start}, {stop} + 1),\n"
            f"{self._nindent}    [&](const int {node.variable.name}) {{\n"
            f"{body}{self._nindent}}});\n"
            f"{self._nindent}team.team_barrier();\n")

    def assignment_node(self, node) -> str:
        """Let one member make a team-level write to an array.

        Every member of the team executes the body of a hierarchical launch,
        so an array element assigned outside any of the region's parallel
        loops is written by all of them at once. The values agree, but
        concurrent writes to one element are a race in the memory model
        whatever they carry, so the statement is wrapped in
        ``Kokkos::single``. A scalar needs nothing: it is a per-member local,
        and each member writing its own copy races with no one.

        Inside a parallel loop the members already hold disjoint iterations,
        so the write is theirs alone and is left as it is. So is every
        assignment in the two flat launch shapes, whose members are cells
        rather than lanes of one.

        An array constructor on the right-hand side writes array elements
        too, and names its target without subscripting it -- ``x = [a, b]``
        -- so it is wrapped for the same reason. All of its element
        assignments go inside one ``Kokkos::single``, which is both cheaper
        than one region each and what the Fortran meant: the statement is one
        assignment.

        :param node: the assignment in the captured body.
        :type node: :py:class:`psyclone.psyir.nodes.Assignment`

        :returns: the assignment, wrapped in ``Kokkos::single`` and followed
            by a barrier where the team would otherwise race.
        :rtype: str
        """
        writes_an_array = isinstance(node.lhs, ArrayReference) or isinstance(
            node.rhs, ArrayConstructor)
        if (not self._parallel_loops or self._parallel_depth
                or not writes_an_array):
            return super().assignment_node(node)

        self._depth += 1
        inner = super().assignment_node(node)
        self._depth -= 1
        return (
            f"{self._nindent}Kokkos::single(Kokkos::PerTeam(team), "
            "[&]() {\n"
            f"{inner}{self._nindent}}});\n"
            f"{self._nindent}team.team_barrier();\n")

    def arrayreference_node(self, node: ArrayReference) -> str:
        """Emit an indexed View access with Fortran lower bounds removed.

        This is the one place a subscript of a described array is written, so
        it is the one place the array's declared origin is applied: no path
        through the writer can reach an element of a View without subtracting
        the offset its description carries. That matters more than it reads.
        An array whose origin the region has right and whose subscripts it has
        wrong compiles, links and runs, and returns the wrong answer.

        The subtraction is emitted whenever the offset is non-empty, which
        includes an origin of ``0``: ``u_e(k - 0)`` states the origin the
        access was written against, where ``u_e(k)`` would be
        indistinguishable from a subscript the offset had never reached.

        A :py:class:`KokkosConstant` is a plain C array rather than a View, so
        it is subscripted with brackets; everything else about the access,
        including the origin, is the same.

        :param node: the array reference in the captured body.

        :returns: the equivalent zero-based View access.

        :raises ValueError: if the region described neither a View nor scratch
            for the array, or if it supplied a different number of index
            offsets than the reference has indices.
        """
        try:
            view = self._views[node.name]
        except KeyError as err:
            raise ValueError(
                f"Array '{node.name}' has no Kokkos View "
                "description.") from err

        if len(node.indices) != len(view.index_offsets):
            raise ValueError(
                f"Array '{node.name}' has {len(node.indices)} kernel indices "
                f"but {len(view.index_offsets)} offsets were supplied.")
        indices = []
        for index, offset in zip(node.indices, view.index_offsets):
            expression = self._visit(index)
            if offset:
                expression = f"({expression} - {offset})"
            indices.append(expression)
        indices.extend(view.extra_indices)
        if isinstance(view, KokkosConstant):
            return f"{node.name}[{indices[0]}]"
        return f"{node.name}({', '.join(indices)})"


__all__ = ["KokkosArrayExpressionMixin"]
