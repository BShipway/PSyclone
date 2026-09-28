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

from psyclone.psyir.backend.kokkos_array_intrinsics import (
    KokkosArrayIntrinsics)
from psyclone.psyir.nodes import (
    ArrayConstructor, Assignment, IntrinsicCall, Literal, Reference, Routine,
    Schedule, WhileLoop)
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

    #: The name of the local an array constructor is assigned to when it
    #: stands inside a larger array expression; numbered by
    #: ``SymbolTable.new_symbol`` as the one above is.
    _CONSTRUCTOR_TEMPORARY = "constructor"

    #: The root name of the local an array-valued intrinsic's operand is
    #: assigned to; the intrinsic's own name goes in front of it.
    _OPERAND_TEMPORARY = "operand"

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
    def _elemental_datatype(operand):
        """Return the type of ``operand``, an elemental call's being an array.

        The PSyIR types an elemental intrinsic by its scalar result:
        ``real(PANEL_ROT_MATRIX(:,:,panel_id), r_double)`` is a
        ``real(r_double)``. Applied to an array, Fortran gives an array of
        that result in the argument's shape, and that is the type returned.

        :param operand: the expression to type.
        :type operand: :py:class:`psyclone.psyir.nodes.DataNode`

        :returns: its type, with an elemental call applied to an array
            given that array's shape.
        :rtype: :py:class:`psyclone.psyir.symbols.DataType`
        """
        datatype = operand.datatype
        if not (isinstance(operand, IntrinsicCall) and operand.is_elemental
                and isinstance(datatype, ScalarType)):
            return datatype
        for argument in operand.arguments:
            if isinstance(argument.datatype, ArrayType):
                return ArrayType(datatype, argument.datatype.shape)
        return datatype

    @staticmethod
    def _is_array_expression(actual, datatype=None):
        """Return whether ``actual`` is an expression with a declarable shape.

        :param actual: the actual argument to judge.
        :type actual: :py:class:`psyclone.psyir.nodes.DataNode`
        :param datatype: the type to judge it by, where the caller knows it
            better than ``actual.datatype`` does; that one by default.
        :type datatype: Optional[:py:class:`psyclone.psyir.symbols.DataType`]

        :returns: whether it is neither a variable nor a literal, and its
            type is an array of a known intrinsic type whose every extent is
            given by bounds.
        :rtype: bool
        """
        if isinstance(actual, (Reference, Literal)):
            return False
        datatype = datatype or actual.datatype
        if not isinstance(datatype, ArrayType):
            return False
        if not isinstance(datatype.intrinsic, ScalarType.Intrinsic):
            return False
        return all(isinstance(extent, ArrayType.ArrayBounds)
                   for extent in datatype.shape)

    @classmethod
    def _hoist_constructors(cls, schedule):
        """Assign each array constructor inside an array expression to a local.

        LFRic's ``alphabetar2xyz`` writes ``xyz = radius / rho * (/ 1.0,
        tan(alpha), tan(beta) /)``.
        :py:class:`~psyclone.psyir.transformations.ArrayAssignment2LoopsTrans`
        declines any assignment holding a constructor, since the loop it
        writes would subscript the constructor, and no writer can render
        that. A constructor standing alone on the right is what the backend
        already writes element by element, so the constructor is given that
        shape: it is assigned to a new local immediately before the
        statement, and the statement reads the local, which the lowering
        subscripts like any other array.

        A constructor is left as written when any of its elements is itself
        an array, or its element type is not a known intrinsic one, since
        then its length or its declaration cannot be stated; the lowering
        then refuses the statement in its own words.

        :param schedule: the kernel schedule to prepare.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        """
        table = schedule.symbol_table
        for constructor in schedule.walk(ArrayConstructor):
            statement = constructor.ancestor(Assignment)
            if statement is None or statement.rhs is constructor:
                continue
            # pylint: disable-next=no-member
            if not cls._is_array_valued(statement):
                continue
            element = constructor.datatype.elemental_type
            if not isinstance(element, ScalarType) or any(
                    not isinstance(child.datatype, ScalarType)
                    for child in constructor.children):
                continue
            local = table.new_symbol(
                cls._CONSTRUCTOR_TEMPORARY, symbol_type=DataSymbol,
                datatype=ArrayType(element, [len(constructor.children)]))
            statement.parent.children.insert(
                statement.position,
                Assignment.create(Reference(local), constructor.copy()))
            constructor.replace_with(Reference(local))

    @classmethod
    def _hoist_intrinsic_operands(cls, schedule):
        """Assign each expression an array-valued intrinsic reads to a local.

        :py:class:`~psyclone.psyir.backend.kokkos_array_intrinsics.\
KokkosArrayIntrinsics` writes ``MATMUL``, ``DOT_PRODUCT``, the folds and
        the maps as loops that subscript each operand, so an operand has to
        be something with subscripts: a whole array, a section, or another
        intrinsic of the tier. LFRic's ``alphabetar2xyz`` reads
        ``matmul(real(PANEL_ROT_MATRIX(:,:,panel_id), r_double), xyz)``,
        whose first operand is none of them. As
        :py:meth:`_hoist_array_expressions` does for an actual, the operand
        is evaluated into a new local immediately before the statement,
        where Fortran would have evaluated it into a temporary anyway, and
        the intrinsic reads the local. The assignment made is array-valued
        and is lowered to a loop with every other.

        An operand whose shape or element type the PSyIR cannot state is
        left as written, for the writer to refuse; and so is one in the
        condition of a ``DO WHILE``, for the reason
        :py:meth:`_hoist_array_expressions` gives.

        The calls are visited innermost first. A hoisted operand is moved,
        so an intrinsic nested in it, as LFRic's ``compute_total_pv`` nests
        ``matmul`` in a sum ``dot_product`` reads, has to have been prepared
        already: visited afterwards it would no longer be in the schedule.

        :param schedule: the kernel schedule to prepare.
        :type schedule: :py:class:`psyclone.psyir.nodes.KernelSchedule`
        """
        table = schedule.symbol_table
        for call in reversed(schedule.walk(IntrinsicCall)):
            if not KokkosArrayIntrinsics.handles(call):
                continue
            statement = call
            while not isinstance(statement.parent, Schedule):
                statement = statement.parent
            if isinstance(statement, WhileLoop):
                continue
            for operand in list(call.arguments):
                if KokkosArrayIntrinsics.handles(operand):
                    continue
                datatype = cls._elemental_datatype(operand)
                if not cls._is_array_expression(operand, datatype):
                    continue
                local = table.new_symbol(
                    f"{call.intrinsic.name.lower()}_"
                    f"{cls._OPERAND_TEMPORARY}",
                    symbol_type=DataSymbol, datatype=datatype)
                statement.parent.children.insert(
                    statement.position,
                    Assignment.create(Reference(local), operand.copy()))
                operand.replace_with(Reference(local))


__all__ = ["LFRicKokkosTemporaryMixin"]
