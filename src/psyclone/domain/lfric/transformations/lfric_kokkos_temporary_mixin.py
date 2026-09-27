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


"""Give an array expression passed to a callee a variable to live in.

The class here is a **mixin**, inherited by
:py:class:`~psyclone.domain.lfric.transformations.LFRicKokkosTrans` rather
than instantiated. It holds no instance state and every method is a
``classmethod`` or a ``staticmethod``, which is what makes the mixin sound: it
is a namespace with an inheritable ``cls``, not an object. It sits beside
:py:class:`~psyclone.domain.lfric.transformations.lfric_kokkos_bound_mixin.\
LFRicKokkosBoundMixin`, whose :py:meth:`_inline_one` prepares a call by
calling :py:meth:`~LFRicKokkosTemporaryMixin._hoist_array_expressions` among
the other preparation steps.

**What Fortran does with such an actual.** A call may pass an expression
where the callee declares an array -- LFRic's native Jacobian passes
``chi_3_df+radius`` to ``jacobian_abr2XYZ``, which declares
``radius(nlayers)`` and reads ``radius(:)``. Fortran evaluates the
expression into a temporary before the call, and the callee reads that.
:py:class:`~psyclone.psyir.transformations.InlineTrans` has no such step:
it puts the actual where the dummy stood, which for an array read by
subscript means subscripting the expression, and it fails part-way through
with an ``AttributeError`` rather than refusing.

**What this does instead.** It writes the temporary out: the expression is
assigned to a new local array immediately before the statement holding the
call, and the call is passed the local. That is the evaluation Fortran made
anyway, now as a statement the inliner reads like any other, and the local
is an automatic array of the expression's own shape, which the rest of the
transformation already knows how to place.

**Where it does not.** An expression whose shape the PSyIR cannot state in
bounds, or whose element type it does not know, is left as the kernel wrote
it, since there is no declaration to give the local; and so is one inside
the condition of a ``DO WHILE``, where an assignment before the loop would be
evaluated once and Fortran evaluates the condition on every trip. Either is
then refused by ``InlineTrans`` in its own words, with the call named.
"""

from psyclone.psyir.nodes import (
    Assignment, Literal, Reference, Routine, Schedule, WhileLoop)
from psyclone.psyir.symbols import ArrayType, DataSymbol, ScalarType


class LFRicKokkosTemporaryMixin:
    """Assign each array-valued expression actual to a local first."""
    # A mixin contributing only private helpers has none of its own by
    # design; the class it is mixed into carries the public interface.
    # pylint: disable=too-few-public-methods

    #: The root name of the local an expression is assigned to. The callee's
    #: name goes in front of it, so a reader of the generated region meets
    #: ``jacobian_abr2xyz_actual`` where the Fortran passed an expression, and
    #: ``SymbolTable.new_symbol`` numbers a second one rather than colliding
    #: with the first or with anything the kernel declared.
    _ACTUAL_TEMPORARY = "actual"

    @classmethod
    def _hoist_array_expressions(cls, call):
        """Assign each array-valued expression ``call`` passes to a local.

        Made before
        :py:class:`~psyclone.psyir.transformations.InlineTrans` is asked to
        substitute the actuals for the dummies, so that every array actual
        it meets is a variable.

        :param call: the call to prepare.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        """
        statement = call
        while not isinstance(statement.parent, Schedule):
            statement = statement.parent
        if isinstance(statement, WhileLoop):
            return
        # pylint: disable=no-member
        name = cls._callee_name(call).lower()
        table = call.ancestor(Routine).symbol_table
        for actual in list(call.arguments):
            if not cls._is_array_expression(actual):
                continue
            local = table.new_symbol(
                f"{name}_{cls._ACTUAL_TEMPORARY}", symbol_type=DataSymbol,
                datatype=actual.datatype)
            statement.parent.children.insert(
                statement.position,
                Assignment.create(Reference(local), actual.copy()))
            actual.replace_with(Reference(local))

    @staticmethod
    def _is_array_expression(actual):
        """Return whether ``actual`` is an expression with a declarable shape.

        :param actual: the actual argument to judge.
        :type actual: :py:class:`psyclone.psyir.nodes.DataNode`

        :returns: whether it is neither a variable nor a literal, and its
            type is an array of a known intrinsic type whose every extent is
            given by bounds.
        :rtype: bool
        """
        if isinstance(actual, (Reference, Literal)):
            return False
        datatype = actual.datatype
        if not isinstance(datatype, ArrayType):
            return False
        if not isinstance(datatype.intrinsic, ScalarType.Intrinsic):
            return False
        return all(isinstance(extent, ArrayType.ArrayBounds)
                   for extent in datatype.shape)


__all__ = ["LFRicKokkosTemporaryMixin"]
