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

What the region takes for each of the kernel's arguments, what the PSy layer
passes in its place and what the ``bind(C)`` interface declares are not here
but in ``LFRicKokkosArgumentMixin``, so that one module holds the whole of the
correspondence between a formal and its actual.

The one constraint that follows is that a method reaching a helper of the
sibling mixins ``LFRicKokkosTypesMixin`` and ``LFRicKokkosBoundsMixin`` does
so through ``cls``, resolved on ``LFRicKokkosTrans``. Calling such a method
directly on any of them is therefore not supported, and
:py:meth:`LFRicKokkosCallMixin._local_arrays` does reach across: it asks
``cls._c_type``, ``cls._extents`` and ``cls._origins``.
"""

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

    @classmethod
    def _local_arrays(cls, schedule):
        """Describe the kernel's automatic arrays as team scratch.

        Each one becomes a scratch View private to the team rank running the
        cell, which is what makes the region's per-cell temporaries per-cell.
        A scalar local needs none of this and is declared in the region body,
        so only arrays appear here.

        :py:meth:`_validate_locals` has already refused anything this could
        not describe.

        :param schedule: the kernel schedule being captured.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`

        :returns: one description per automatic array, in declaration order.
        :rtype: tuple[
            :py:class:`psyclone.psyir.backend.kokkos.KokkosScratch`, ...]
        """
        scratch = []
        for symbol in schedule.symbol_table.automatic_datasymbols:
            if not symbol.is_array:
                continue
            extents = cls._extents(symbol)
            scratch.append(KokkosScratch(
                symbol.name, cls._c_type(symbol), extents,
                index_offsets=cls._origins(symbol)))
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
