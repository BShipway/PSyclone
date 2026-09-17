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

"""Name the region's scratch, its counter and the imports its call needs.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather than
instantiated. It holds no instance state and every method is a
``staticmethod`` or a ``classmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object.

What the region takes for each of the kernel's arguments and what the PSy
layer passes in its place are not here but in ``LFRicKokkosArgumentMixin``, so
that one module holds the whole of the correspondence between a formal and its
actual; what the ``bind(C)`` interface then declares for each of them is
``LFRicKokkosInterfaceMixin``.

The one constraint that follows is that a method reaching a helper of the
sibling mixins ``LFRicKokkosTypesMixin`` and ``LFRicKokkosBoundsMixin`` does
so through ``cls``, resolved on ``LFRicKokkosTrans``. Calling such a method
directly on any of them is therefore not supported, and
:py:meth:`LFRicKokkosCallMixin._local_arrays` does reach across: it asks
``cls._c_type``, ``cls._extents`` and ``cls._origins``.
"""

import ast

from psyclone.psyir.backend.kokkos import KokkosScratch
from psyclone.psyir.nodes import Loop, Reference
from psyclone.psyir.symbols import (
    ContainerSymbol, DataSymbol, ImportInterface, UnresolvedType)


class LFRicKokkosCallMixin:
    """Hold what the call site needs beyond the arguments themselves.

    Three questions: what scratch the region reserves, what becomes of the
    loop counter the replaced loop was counting with, and how a module
    constant the body reads is imported into the PSy layer. What the region
    takes for each kernel argument, and what the PSy layer passes for it, is
    ``LFRicKokkosArgumentMixin``; what a symbol is in C terms is
    ``LFRicKokkosTypesMixin``.
    """
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    @staticmethod
    def _drop_unused_counter(routine, symbol):
        """Undeclare the replaced loop's counter if nothing else counts by it.

        The PSy layer declares one counter per iteration space, so an invoke
        whose only cell loop is captured is left declaring a variable it
        never mentions again -- which LFRic compiles with
        ``-Werror=unused-variable``. An invoke with a second cell loop keeps
        it, which is why this asks rather than assumes.

        :param routine: the PSy-layer routine holding the replaced loop.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param symbol: the counter the replaced loop was counting with.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        table = routine.symbol_table
        if table.lookup(symbol.name, otherwise=None) is not symbol:
            return
        if symbol in table.argument_list:
            return
        if any(reference.symbol is symbol
               for reference in routine.walk(Reference)):
            return
        # A Loop holds its control variable as an attribute rather than as a
        # Reference, so the walk above would not see a sibling loop still
        # counting with it.
        if any(loop.variable is symbol for loop in routine.walk(Loop)):
            return
        table.remove(symbol)

    #: The most elements a kernel-local array may hold and still be given
    #: to every member of the team rather than shared in team scratch. Every
    #: member holding its own copy costs the team ``members * elements``
    #: where sharing costs ``elements``, so the cap is what keeps the cure
    #: from being worse than the disease: a thirty-level column replicated
    #: across thirty-two members is not a saving. Sixteen has headroom over
    #: the largest constant-shaped local in GungHo, which holds nine (the
    #: ``3, 3`` coefficient array of ``polyv_wtheta_koren``); the column
    #: arrays, which are the ones worth refusing, are shaped from
    #: ``nlayers`` or a number of dofs and are refused by
    #: :py:meth:`_member_local_size` before the cap is reached.
    MEMBER_LOCAL_MAX_ELEMENTS = 16

    #: The most dimensions such an array may have, which is the number the
    #: generated wrapper subscripts:
    #: :py:func:`psyclone.psyir.backend.kokkos_launch.member_local_definition`
    #: emits three ``operator()`` overloads and no more.
    MEMBER_LOCAL_MAX_RANK = 3

    #: How the arithmetic of a literal extent is evaluated, which is the
    #: whole of what :py:meth:`_literal_value` knows how to do. Division
    #: truncates toward zero, as C and Fortran both truncate an integer
    #: quotient, and answers nothing where the divisor is zero.
    _LITERAL_OPERATORS = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: int(left / right) if right else None,
    }

    @classmethod
    def _member_local_size(cls, extents):
        """Return how many elements a shape holds, if that is a constant.

        The extents are C expressions by the time they are here, so this
        reads them as arithmetic over integer literals -- ``2``, ``(2 + 1)``
        -- and answers ``None`` for anything else. An extent naming a scalar
        of the region, ``nlayers`` or ``ndf_w3``, is precisely what it must
        answer ``None`` for: that size is not known until the launch, and
        storage the generated C++ cannot size at compile time cannot be a
        member's own.

        Division truncates toward zero, as both Fortran and C++ do, so the
        count agrees with the one the scratch View would have been given.

        :param extents: the C extent expressions of one array.
        :type extents: tuple[str, ...]

        :returns: the product of the extents, or ``None`` where any of them
            is not constant.
        :rtype: Optional[int]
        """
        total = 1
        for extent in extents:
            try:
                tree = ast.parse(extent, mode="eval").body
            except SyntaxError:
                return None
            value = cls._literal_value(tree)
            if value is None:
                return None
            total *= value
        return total

    @classmethod
    def _literal_value(cls, node):
        """Return the value of a literal arithmetic expression, or ``None``.

        Named for what it answers rather than ``_fold``, which is taken:
        every method here lands in ``LFRicKokkosTrans``'s namespace, and
        ``LFRicKokkosConstantsMixin._fold`` already folds a PSyIR expression
        over the kernel's named constants there. A sibling mixin's helper is
        silently overridden by one of the same name, and the override fails
        nowhere near its cause -- this one was found as a declared bound
        that stopped folding, two mixins away.

        :param node: the parsed expression.
        :type node: :py:class:`ast.expr`

        :returns: the integer the expression evaluates to, or ``None`` where
            it names anything, is not integer, or is an operation this does
            not evaluate.
        :rtype: Optional[int]
        """
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, int) else None
        if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)):
            operand = cls._literal_value(node.operand)
            if operand is None or isinstance(node.op, ast.UAdd):
                return operand
            return -operand
        evaluate = cls._LITERAL_OPERATORS.get(type(node.op)) if isinstance(
            node, ast.BinOp) else None
        if evaluate is None:
            return None
        left = cls._literal_value(node.left)
        right = cls._literal_value(node.right)
        if left is None or right is None:
            return None
        return evaluate(left, right)

    @classmethod
    def _is_member_local(cls, symbol, extents, parallel_loops, targets):
        """Say whether every member of the team may hold its own copy.

        A kernel-local array is team scratch because the team shares it. It
        does not have to be, and four conditions together say when it need
        not: **no loop the launch spreads over the team names it**, so every
        member computes the same values into its own copy and no member ever
        reads another's; **its shape is a compile-time constant** small
        enough that a copy each is cheap, both of which
        :py:meth:`_member_local_size` answers; and **no aliasing pointer is
        aimed at it**, since such a pointer is generated as a View handle
        and there would be no View to hand it.

        The first is the correctness condition and the rest are cost and
        capability. What makes the distinction worth drawing is that a
        two-element array of team-uniform integers in team scratch buys
        nothing and costs a ``Kokkos::single`` and a barrier at every write,
        and -- separately, and measured under phase 7's task W7 -- that
        ``nvcc`` 13.3 miscompiles a region holding four of them at any
        optimisation above ``-Xcicc -O1``.

        A region with no spread loops is not asked: its members are whole
        cells rather than lanes of one, so it reserves one array per member
        already and there is nothing to move.

        :param symbol: the automatic array being described.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param extents: its C extent expressions.
        :type extents: tuple[str, ...]
        :param parallel_loops: the loops the launch will spread over the
            team, as ``_parallel_loops`` gives them.
        :type parallel_loops: tuple[
            :py:class:`psyclone.psyir.nodes.Loop`, ...]
        :param targets: the names every aliasing pointer of the body is
            aimed at.
        :type targets: set[str]

        :returns: whether the array is described as per-member storage.
        :rtype: bool
        """
        if not parallel_loops or symbol.name in targets:
            return False
        if not extents or len(extents) > cls.MEMBER_LOCAL_MAX_RANK:
            return False
        size = cls._member_local_size(extents)
        if size is None or not 0 < size <= cls.MEMBER_LOCAL_MAX_ELEMENTS:
            return False
        return not any(
            reference.symbol is symbol
            for loop in parallel_loops
            for reference in loop.walk(Reference))

    @classmethod
    def _local_arrays(cls, schedule, parallel_loops=()):
        """Describe the kernel's automatic arrays as team scratch.

        Each one becomes a scratch View private to the team rank running the
        cell, which is what makes the region's per-cell temporaries per-cell.
        A scalar local needs none of this and is declared in the region body,
        so only arrays appear here, and an aliasing pointer does not: it is a
        second name for storage something else owns and is described by
        :py:meth:`_region_aliases` instead.

        An array :py:meth:`_is_member_local` accepts is described as scratch
        too, and carries
        :py:attr:`~psyclone.psyir.backend.kokkos.KokkosScratch.member_local`,
        which the writer reads as "declare one per member and reserve no
        team scratch for it".

        :py:meth:`_validate_locals` has already refused anything this could
        not describe.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        :param parallel_loops: the loops the launch will spread over the
            team, as ``_parallel_loops`` gives them, and empty for a region
            that spreads none. Passed in rather than computed here because
            the caller has paid for the dependence analysis that answers it.
        :type parallel_loops: tuple[
            :py:class:`psyclone.psyir.nodes.Loop`, ...]

        :returns: one description per automatic array, in declaration order.
        :rtype: tuple[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScratch`, ...]
        """
        scratch = []
        aliases = cls._alias_targets(schedule)
        targets = {target for aimed in aliases.values() for target in aimed}
        for symbol in schedule.symbol_table.automatic_datasymbols:
            if not symbol.is_array:
                continue
            # An alias owns no storage: it is described by
            # ``_region_aliases`` as a handle over an array described here or
            # among the region's arguments, and reserving scratch for it as
            # well would reserve a column nothing ever reads.
            if symbol.name in aliases:
                continue
            extents = cls._extents(symbol)
            scratch.append(KokkosScratch(
                symbol.name, cls._c_type(symbol), extents,
                index_offsets=cls._origins(symbol),
                member_local=cls._is_member_local(
                    symbol, extents, parallel_loops, targets)))
        return tuple(scratch)

    @staticmethod
    def _import_constant(symbol_table, name, container, orig_name=None):
        """Return the PSy-layer import for one kernel module constant.

        Where the kernel renamed the constant on import, the PSy layer has to
        rename it too: it is the module's name that the module declares, and
        the local one that the generated call passes.

        :param symbol_table: the PSy-layer table the import is added to.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        :param str name: the name the generated code knows the constant by.
        :param str container: the module it is imported from.
        :param orig_name: the name the module declares it under, where that
            differs from ``name``, and ``None`` otherwise.
        :type orig_name: Optional[str]

        :returns: the existing symbol if the PSy layer already has one, and a
            new imported symbol otherwise.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        existing = symbol_table.lookup(name, otherwise=None)
        if existing:
            return existing
        module = symbol_table.find_or_create(
            container, symbol_type=ContainerSymbol)
        return symbol_table.find_or_create(
            name, symbol_type=DataSymbol, datatype=UnresolvedType(),
            interface=ImportInterface(module, orig_name=orig_name))


__all__ = ["LFRicKokkosCallMixin"]
