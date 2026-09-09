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

"""The writer's own handlers for the arrays in a captured body.

:py:class:`KokkosArrayExpressionMixin` is the writer side of the array work:
the visitor handlers a captured body reaches when it meets a loop, an
assignment or a subscript, and the point at which each decides whether the
statement is generated as it stands or handed to
:py:class:`~psyclone.psyir.backend.kokkos_array_expression.KokkosArrayExpression`
to be lowered to a nest.

It is a module of its own, and not part of
:py:mod:`~psyclone.psyir.backend.kokkos_array_expression`, because the two
answer to different things. The lowering engine there is a translation from
one array expression to the C++ that computes it, and it grows when a new
shape of expression has to be generated. What is here is the writer's
contract with :py:class:`~psyclone.psyir.backend.visitor.PSyIRVisitor` and
with :py:class:`~psyclone.psyir.backend.c.CWriter`: which node kinds the
Kokkos writer claims, what each hands on by ``super()``, and where the team
rule that makes a write safe is applied. That grows when the *region* gains a
capability rather than when an expression does. Neither reads the other's
internals -- the mixin calls only the engine's public
:py:meth:`~psyclone.psyir.backend.kokkos_array_expression.KokkosArrayExpression.lower`,
:py:meth:`~psyclone.psyir.backend.kokkos_array_expression.KokkosArrayExpression.holds`
-- so the split is along a seam that already existed.

It is written as a mixin rather than as free functions because every one of
its methods is a visitor handler, reached by name from
:py:class:`~psyclone.psyir.backend.visitor.PSyIRVisitor` and falling through
to :py:class:`~psyclone.psyir.backend.c.CWriter` by ``super()``; a handler
that is not on the writer is not called at all. It is mixed into
:py:class:`~psyclone.psyir.backend.kokkos.KokkosWriter` ahead of that C
writer, so that these handlers are found first.
"""

from psyclone.psyir.backend.kokkos_array_expression import (
    KokkosArrayExpression)
from psyclone.psyir.backend.kokkos_constant import KokkosConstant
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import (
    ArrayConstructor, ArrayReference, BinaryOperation, Range, Reference)
from psyclone.psyir.symbols import ArrayType

#: The read-modify-write shapes an atomic answers, as the PSyIR operator
#: joining an element to its own contribution -> the Kokkos function that
#: applies it indivisibly, and whether the element may appear on either side
#: of that operator. ``a = a + x`` and ``a = x + a`` are one update; ``a =
#: a - x`` and ``a = x - a`` are two different ones, and only the first is an
#: ``atomic_sub``, so subtraction and division are matched on the left alone.
#:
#: All four functions are declared by Kokkos for every arithmetic element
#: type. The value converts to the View's element type at the call, because
#: Kokkos deduces the type from the pointer and from nothing else -- so a
#: ``double`` expression accumulated into a single-precision field is rounded
#: once, where a generated ``float`` temporary would have rounded it twice.
ATOMIC_UPDATES = {
    BinaryOperation.Operator.ADD: ("Kokkos::atomic_add", True),
    BinaryOperation.Operator.MUL: ("Kokkos::atomic_mul", True),
    BinaryOperation.Operator.SUB: ("Kokkos::atomic_sub", False),
    BinaryOperation.Operator.DIV: ("Kokkos::atomic_div", False),
}


def atomic_update_operands(assignment):
    """Return the operands of a read-modify-write of the assignment's target.

    The shape recognised is an assignment whose right-hand side joins the
    left-hand side to one other expression by one of the operators in
    :py:data:`ATOMIC_UPDATES`. That is the whole of what an atomic can do:
    anything else naming the target on the right -- two occurrences of it,
    an occurrence under a function call, a different element of it -- is a
    computation the hardware has no single indivisible instruction for.

    Matching is on the PSyIR, not on the field's LFRic metadata, because the
    metadata says only that the argument is shared. ``gh_inc`` is carried by
    kernels that multiply as well as by kernels that add, so what the update
    *is* can only be read from the body.

    :param assignment: the statement to inspect.
    :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`

    :returns: the Kokkos function and the expression contributed to the
        target, or None if the statement is not a recognised update.
    :rtype: Optional[Tuple[str, :py:class:`psyclone.psyir.nodes.Node`]]
    """
    right = assignment.rhs
    if not isinstance(right, BinaryOperation):
        return None
    entry = ATOMIC_UPDATES.get(right.operator)
    if entry is None:
        return None
    function, commutative = entry
    left, other = right.children[0], right.children[1]
    if left == assignment.lhs and not _names(other, assignment.lhs):
        return (function, other)
    if (commutative and other == assignment.lhs
            and not _names(left, assignment.lhs)):
        return (function, left)
    return None


def atomic_store_operand(assignment):
    """Return the value a replacing write stores, if a store can carry it.

    The counterpart of :py:func:`atomic_update_operands` for the other kind
    of sharing: where the cells reaching an element replace it rather than
    contribute to it, the whole right-hand side is what the store writes and
    no shape is required of it. One thing disqualifies it, and it is the rule
    that function applies to the operand of an accumulation: a value naming
    the target reads an element another cell may be storing to at that
    moment, which is a race the store does not answer -- the value is read
    before the store begins and is no part of it.

    :param assignment: the statement to inspect.
    :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`

    :returns: the expression stored into the target, or None if it reads the
        element being replaced.
    :rtype: Optional[:py:class:`psyclone.psyir.nodes.Node`]
    """
    if _names(assignment.rhs, assignment.lhs):
        return None
    return assignment.rhs


def _names(expression, target):
    """Whether an expression reads the array the update is writing.

    The value an atomic is given is read before the update begins and is not
    part of it, so an expression naming the target reads a location another
    cell may be updating at the same moment. That is a race whatever the
    writer generates, and it is the target's own dof for ``acc + acc`` and a
    neighbouring one for ``acc(i) + acc(j)``; neither is answered by making
    the write indivisible.

    Matching is on the symbol rather than on the subscript, because two
    subscripts that differ textually may still be the same dof: which they
    are is decided by a dofmap the writer cannot read.

    :param expression: the operand contributed to the target.
    :type expression: :py:class:`psyclone.psyir.nodes.Node`
    :param target: the assignment's left-hand side.
    :type target: :py:class:`psyclone.psyir.nodes.Reference`

    :returns: whether the expression refers to the target's array.
    :rtype: bool
    """
    return any(reference.symbol is target.symbol
               for reference in expression.walk(Reference))


class KokkosArrayExpressionMixin:
    """The writer's handling of arrays in a captured body.

    Mixed into :py:class:`~psyclone.psyir.backend.kokkos.KokkosWriter` ahead
    of :py:class:`~psyclone.psyir.backend.c.CWriter`, so that these handlers
    are found first and reach the C writer's by ``super()``. The state they
    read -- ``_views``, ``_kind_types``, ``_parallel_loops``,
    ``_parallel_depth`` and ``_private_scalars`` -- is established by the
    writer for the duration of one region and is documented there.
    """

    #: How many of the region's chosen loops the visitor is currently
    #: inside, and how far the generated source is indented. Both are
    #: counters the writer owns and both start at zero, so the mixin declares
    #: that starting value rather than annotating a name it never sets: a
    #: counter read before its owner has established it is at zero either
    #: way, and pylint can only check an attribute the class states.
    _depth = 0
    _parallel_depth = 0
    #: The scalars each spread loop declares inside its own lambda, keyed by
    #: the ``id`` of the loop. Empty until the writer establishes it, which
    #: is the answer for a region with no loop-private scalar in any case.
    _private_scalars = {}
    #: The lowering this writer is using, created on first use and discarded
    #: with the region, because the names it generates are numbered and a
    #: second region must start again from zero.
    _array_expressions = None

    @property
    def array_expressions(self):
        """Return the lowering that turns an array expression into a nest.

        :returns: the lowering, created on first use for this region.
        :rtype: :py:class:`KokkosArrayExpression`
        """
        if self._array_expressions is None:
            self._array_expressions = KokkosArrayExpression(self)
        return self._array_expressions

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

        The lambda opens with a declaration of every scalar this loop was
        found to own -- the values it works in and the counters of the loops
        nested inside it -- so that each iteration holds its own and no
        statement after the loop can read what an iteration left behind. Once
        the members share the iterations out, the value each member is left
        holding is the one from the last iteration it happened to run, which
        is a value nothing should be reading.
        Which those are is
        :py:func:`~psyclone.psyir.backend.kokkos_team_scalars.team_private_scalars`'s
        answer, settled before any of this was generated; a loop writing a
        scalar that cannot be made private never reaches here, because the
        region carrying it is refused.

        A ``team_barrier`` follows unconditionally. A statement after the loop
        may read an *array element* the loop wrote, and working out whether
        one does is a second dependence analysis this writer does not perform;
        under ``Kokkos::AUTO`` on the OpenMP backend the team has one member
        and the barrier costs nothing measurable.

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
        declarations = "".join(
            self.gen_local_variable(symbol)
            for symbol in self._private_scalars.get(id(node), ()))
        body = "".join(self._visit(child) for child in node.loop_body)
        self._depth -= 1
        self._parallel_depth -= 1
        return (
            f"{self._nindent}Kokkos::parallel_for("
            f"Kokkos::TeamVectorRange(team, {start}, {stop} + 1),\n"
            f"{self._nindent}    [&](const int {node.variable.name}) {{\n"
            f"{declarations}{body}{self._nindent}}});\n"
            f"{self._nindent}team.team_barrier();\n")

    def _atomic_update(self, node, lowered):
        """Return the atomic call this assignment needs, if it needs one.

        Two kinds of sharing reach here and they are answered differently.
        Where the cells sharing an element *contribute* to it the statement
        has to be a read-modify-write by one of the operators in
        :py:data:`ATOMIC_UPDATES`, because combining the contributions is the
        whole of what the atomic is for. Where they *replace* it -- a View
        the region describes with
        :py:attr:`~psyclone.psyir.backend.kokkos.KokkosView.atomic_store` --
        a plain assignment is answered too, by ``Kokkos::atomic_store``: the
        element is written whole, so no reader sees a value neither cell
        stored. Which cell wrote last is settled by neither, and for a
        replacing write it is the kernel that promises it does not matter.

        A replacing write whose value reads the element it is replacing is
        refused all the same. The value is read before the store begins and
        is no part of it, so the read races with another cell's store however
        the store itself is generated -- the same rule
        :py:func:`atomic_update_operands` applies to the operand of an
        accumulation.

        :param node: the assignment in the captured body.
        :type node: :py:class:`psyclone.psyir.nodes.Assignment`
        :param bool lowered: whether the statement is an array expression
            that :py:class:`KokkosArrayExpression` will lower to a nest.

        :returns: the Kokkos function and the expression contributed to or
            stored into the target, or None if the target is not shared
            between cells.
        :rtype: Optional[Tuple[str, :py:class:`psyclone.psyir.nodes.Node`]]

        :raises VisitorError: if the target is shared but the statement is
            not a read-modify-write an atomic can carry out, or replaces the
            element with a value that reads it, or is a whole section rather
            than one element.
        """
        target = node.lhs
        if not isinstance(target, ArrayReference):
            return None
        view = self._views.get(target.name)
        if not getattr(view, "atomic", False):
            return None
        if lowered:
            raise VisitorError(
                f"Kokkos region updates shared array '{target.name}' with a "
                "whole-array expression, which no single atomic carries out. "
                "Capture it as an element assignment or colour the loop.")
        operands = atomic_update_operands(node)
        if operands is not None:
            return operands
        if getattr(view, "atomic_store", False):
            value = atomic_store_operand(node)
            if value is None:
                raise VisitorError(
                    f"Kokkos region replaces an element of shared array "
                    f"'{target.name}' with a value that reads it, which is a "
                    "race no atomic store answers. Colour the loop instead.")
            return ("Kokkos::atomic_store", value)
        shapes = ", ".join(
            sorted(name for name, _ in ATOMIC_UPDATES.values()))
        raise VisitorError(
            f"Kokkos region writes shared array '{target.name}' with a "
            "statement that is not one of the read-modify-write shapes "
            f"an atomic answers ({shapes}). Colour the loop instead.")

    def _atomic_statement(self, node, function, value):
        """Return one indivisible update as a generated statement.

        The target is passed by address, which is how Kokkos names the
        element to update and also why the update's type is the View's: the
        value's type takes no part in the deduction, so a contribution
        computed more widely than the field is rounded once, at the call.

        :param node: the assignment being generated.
        :type node: :py:class:`psyclone.psyir.nodes.Assignment`
        :param str function: the ``Kokkos::atomic_*`` to call.
        :param value: the expression contributed to the target.
        :type value: :py:class:`psyclone.psyir.nodes.Node`

        :returns: the call, indented as a statement of the captured body.
        :rtype: str
        """
        return (f"{self._nindent}{function}(&{self._visit(node.lhs)}, "
                f"{self._visit(value)});\n")

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

        An assignment holding a :py:class:`~psyclone.psyir.nodes.Range` is no
        statement at all in C++, so it is not generated as one:
        :py:class:`KokkosArrayExpression` lowers it to a nest first, and the
        nest is then wrapped by exactly the rule above. The team-level rule
        applies to the nest as a whole rather than to each of its element
        assignments, for the same reason it applies to a constructor: the
        Fortran statement is one assignment.

        An assignment that updates an element of a View the region marks
        atomic is generated as a ``Kokkos::atomic_*`` call instead, by
        :py:meth:`_atomic_update`. That is a different race from the one
        above and is answered separately: the team rule protects the members
        of one team from each other, and the atomic protects the cells of the
        whole launch, which are on different teams and share a dof. Both
        apply where both are needed -- an atomic executed redundantly by
        every member of a team would accumulate the contribution once per
        member -- so the atomic is what goes inside the ``Kokkos::single``.

        :param node: the assignment in the captured body.
        :type node: :py:class:`psyclone.psyir.nodes.Assignment`

        :returns: the assignment, wrapped in ``Kokkos::single`` and followed
            by a barrier where the team would otherwise race.
        :rtype: str

        :raises VisitorError: if an atomic View is written by a statement no
            atomic can carry out.
        """
        writes_an_array = isinstance(node.lhs, ArrayReference) or isinstance(
            node.rhs, ArrayConstructor)
        # An array constructor's values are positional, and the C writer
        # spreads them over the destination itself; a nest would have to
        # subscript the constructor, which nothing can render.
        # A reduction is lowered even where the statement has no section in
        # it at all: ``x = dot_product(p, q)`` writes a scalar, and still
        # needs the loops that accumulate it written ahead of the assignment.
        lowered = (bool(node.walk(Range))
                   or self.array_expressions.holds(node.rhs)) and not \
            isinstance(node.rhs, ArrayConstructor)
        atomic = self._atomic_update(node, lowered)
        if (not self._parallel_loops or self._parallel_depth
                or not writes_an_array):
            if atomic:
                return self._atomic_statement(node, *atomic)
            if lowered:
                return self.array_expressions.lower(node.rhs, node.lhs)
            return super().assignment_node(node)

        self._depth += 1
        if atomic:
            inner = self._atomic_statement(node, *atomic)
        elif lowered:
            inner = self.array_expressions.lower(node.rhs, node.lhs)
        else:
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
