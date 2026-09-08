# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2019-2026, Science and Technology Facilities Council
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
# Author S. Siso, STFC Daresbury Lab.
# Modified by: J. Henrichs, Bureau of Meteorology
#              A. R. Porter, R. W. Ford and N. Nobre, STFC Daresbury Lab
#              A. B. G. Chalk, STFC Daresbury Lab


'''How the C backend spells a PSyIR operation or intrinsic.

Three of :py:class:`~psyclone.psyir.backend.c.CWriter`'s handlers --
``unaryoperation_node``, ``binaryoperation_node`` and ``intrinsiccall_node``
-- do the same thing and nothing else: look the node up in a table of C
spellings, refuse it if it is not there, and apply the formatter the table
names. They are gathered here, in a mixin the writer inherits, so that the
part of the backend that grows one table entry at a time is separate from the
part that renders statements, declarations and control flow.

Operators and intrinsics are one table and not two, however Fortran spells
them. ``MOD`` is an intrinsic that becomes C's ``%`` operator, ``**`` is an
operator that becomes the ``pow`` function or, for a small integer exponent,
the multiplication tree of
:py:mod:`psyclone.psyir.backend.c_integer_power`, and ``ABS``, ``MOD``,
``MAX`` and ``MIN`` are chosen by the argument's type through
:py:data:`REAL_INTRINSIC_ALTERNATIVES` because Fortran overloads on it and C
does not. Splitting operators from intrinsics would put both halves of each
of those decisions in different modules.

Everything here is inherited rather than instantiated: the mixin holds no
state, and reads only ``self._visit``, which
:py:class:`~psyclone.psyir.backend.visitor.PSyIRVisitor` provides.
'''
from psyclone.psyir.backend.c_integer_power import integer_power
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import (
    BinaryOperation, IntrinsicCall, UnaryOperation)
from psyclone.psyir.symbols import ScalarType


#: Intrinsics whose C spelling depends on the argument's type. Fortran
#: overloads on it and C does not, so a single map entry is a wrong
#: answer for one of the two: ``abs`` binds ``::abs(int)`` and truncates
#: a real, and ``%`` does not compile for one.
REAL_INTRINSIC_ALTERNATIVES = {
    IntrinsicCall.Intrinsic.ABS: "fabs",
    IntrinsicCall.Intrinsic.MOD: "fmod",
    IntrinsicCall.Intrinsic.MAX: "fmax",
    IntrinsicCall.Intrinsic.MIN: "fmin",
    }


def _is_real_argument(node):
    '''Whether an intrinsic's argument is known to be of real type.

    Deliberately answers "no" rather than raising, for every reason it
    might not know: an :py:class:`UnresolvedType`, an
    :py:class:`UnsupportedFortranType`, or a ``datatype`` property that
    raises on a tree the caller assembled by hand. The kind-blind default
    path has to stay reachable, because a caller probing the writer with
    synthetic arguments is asking which intrinsics it supports rather than
    what one particular expression is.

    :param node: the argument to inspect.
    :type node: :py:class:`psyclone.psyir.nodes.DataNode`

    :returns: whether its datatype is a real scalar.
    :rtype: bool

    '''
    try:
        datatype = node.datatype
    except Exception:                            # pylint: disable=W0703
        return False
    return (isinstance(datatype, ScalarType) and
            datatype.intrinsic == ScalarType.Intrinsic.REAL)


class CIntrinsicsMixin:
    '''Spell PSyIR's operators and intrinsics as C.

    Inherited by :py:class:`~psyclone.psyir.backend.c.CWriter` ahead of
    :py:class:`~psyclone.psyir.backend.language_writer.LanguageWriter`, so
    that these handlers are the ones the visitor finds. Nothing here calls
    ``super()``: an operator or intrinsic the tables do not hold is refused,
    because there is no more general writer to fall through to.
    '''

    #: The function a power falls back to when
    #: :py:mod:`psyclone.psyir.backend.c_integer_power` does not write it as
    #: a product tree. C's ``pow`` takes and returns ``double`` whatever it is
    #: given, which is the right answer for C and the wrong one for a
    #: back-end whose operands carry a Fortran kind: a single-precision
    #: ``x ** y`` is computed in double and rounded back, where gfortran calls
    #: ``powf`` and rounds once. A back-end that can spell a power at its
    #: operands' own width overrides this; the Kokkos one does.
    _POW_FUNCTION = "pow"

    def unaryoperation_node(self, node):
        '''This method is called when a UnaryOperation instance is found in
        the PSyIR tree.

        :param node: A UnaryOperation PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.UnaryOperation`

        :returns: The C code as a string.
        :rtype: str

        :raises VisitorError: If this node has more than one child.
        :raises NotImplementedError: If the operator is not supported by the \
            C backend.

        '''
        if len(node.children) != 1:
            raise VisitorError(
                f"UnaryOperation malformed or incomplete. It should "
                f"have exactly 1 child, but found {len(node.children)}.")

        def operator_format(operator_str, expr_str):
            '''
            :param str operator_str: String representing the operator.
            :param str expr_str: String representation of the operand.

            :returns: C language operator expression.
            :rtype: str
            '''
            return "(" + operator_str + expr_str + ")"

        # Define a map with the operator string and the formatter function
        # associated with each UnaryOperation.Operator
        opmap = {
            UnaryOperation.Operator.MINUS: ("-", operator_format),
            UnaryOperation.Operator.PLUS: ("+", operator_format),
            UnaryOperation.Operator.NOT: ("!", operator_format),
            }

        # If the instance operator exists in the map, use its associated
        # operator and formatter to generate the code, otherwise raise
        # an Error.
        try:
            opstring, formatter = opmap[node.operator]
        except KeyError as err:
            raise NotImplementedError(
                f"The C backend does not support the '{node.operator}' "
                f"operator.") from err

        return formatter(opstring, self._visit(node.children[0]))

    def binaryoperation_node(self, node):
        '''This method is called when a BinaryOperation instance is found in
        the PSyIR tree.

        :param node: A BinaryOperation PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.BinaryOperation`

        A power whose exponent is a small integer literal is written as the
        product tree gfortran builds for it rather than as 'pow', so that the
        generated region rounds as the Fortran it replaced;
        :py:mod:`psyclone.psyir.backend.c_integer_power` says which exponents
        and why.

        :returns: The C code as a string.
        :rtype: str

        :raises VisitorError: If this node has fewer children than expected.
        :raises NotImplementedError: If the operator is not supported by the \
            C backend.

        '''
        if len(node.children) != 2:
            raise VisitorError(
                f"BinaryOperation malformed or incomplete. It should "
                f"have exactly 2 children, but found {len(node.children)}.")

        def operator_format(operator_str, expr1, expr2):
            '''
            :param str operator_str: String representing the operator.
            :param str expr1: String representation of the LHS operand.
            :param str expr2: String representation of the RHS operand.

            :returns: C language operator expression.
            :rtype: str
            '''
            return "(" + expr1 + " " + operator_str + " " + expr2 + ")"

        def function_format(function_str, expr1, expr2):
            '''
            :param str function_str: Name of the function.
            :param str expr1: String representation of the first operand.
            :param str expr2: String representation of the second operand.

            :returns: C language binary function expression.
            :rtype: str
            '''
            return function_str + "(" + expr1 + ", " + expr2 + ")"

        # Define a map with the operator string and the formatter function
        # associated with each BinaryOperation.Operator
        opmap = {
            BinaryOperation.Operator.ADD: ("+", operator_format),
            BinaryOperation.Operator.SUB: ("-", operator_format),
            BinaryOperation.Operator.MUL: ("*", operator_format),
            BinaryOperation.Operator.DIV: ("/", operator_format),
            # Reached only by a power the tree above did not write: a
            # non-literal or real exponent, or one out of range. The name is
            # :py:attr:`_POW_FUNCTION` rather than a literal because a
            # back-end that knows its operands' width spells it otherwise.
            BinaryOperation.Operator.POW: (self._POW_FUNCTION,
                                           function_format),
            BinaryOperation.Operator.EQ: ("==", operator_format),
            BinaryOperation.Operator.NE: ("!=", operator_format),
            BinaryOperation.Operator.LT: ("<", operator_format),
            BinaryOperation.Operator.LE: ("<=", operator_format),
            BinaryOperation.Operator.GT: (">", operator_format),
            BinaryOperation.Operator.GE: (">=", operator_format),
            BinaryOperation.Operator.AND: ("&&", operator_format),
            BinaryOperation.Operator.OR: ("||", operator_format),
            }

        # A constant integer power is the multiplications gfortran makes
        # rather than a call to 'pow', which rounds differently; see
        # :py:mod:`psyclone.psyir.backend.c_integer_power`.
        if node.operator == BinaryOperation.Operator.POW:
            product = integer_power(self._visit(node.children[0]),
                                    node.children[1],
                                    _is_real_argument(node.children[0]))
            if product is not None:
                return product

        # If the instance operator exists in the map, use its associated
        # operator and formatter to generate the code, otherwise raise
        # an Error.
        try:
            opstring, formatter = opmap[node.operator]
        except KeyError as err:
            raise VisitorError(
                f"The C backend does not support the '{node.operator}' "
                f"operator.") from err

        return formatter(opstring,
                         self._visit(node.children[0]),
                         self._visit(node.children[1]))

    def intrinsiccall_node(self, node):
        '''This method is called when an IntrinsicCall node is found in
        the PSyIR tree.

        :param node: An IntrinsicCall PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :returns: The C code as a string.
        :rtype: str

        '''
        def binary_operator_format(operator_str, expr_str):
            '''
            :param str operator_str: String representing the operator.
            :param List[str] expr_str: String representation of the operands.

            :returns: C language operator expression.
            :rtype: str

            :raise VisitorError: unexpected number of children.
            '''
            if len(expr_str) != 2:
                raise VisitorError(
                    f"The C Writer binary_operator formatter for IntrinsicCall"
                    f" only supports intrinsics with 2 children, but found "
                    f"'{operator_str}' with '{len(expr_str)}' children.")
            return f"({expr_str[0]} {operator_str} {expr_str[1]})"

        def function_format(function_str, expr_str):
            '''
            :param str function_str: Name of the function.
            :param List[str] expr_str: String representation of the operands.

            :returns: C language unary function expression.
            :rtype: str
            '''
            return function_str + "(" + ", ".join(expr_str) + ")"

        def cast_format(type_str, expr_str):
            '''
            :param str type_str: Name of the new type.
            :param List[str] expr_str: String representation of the operands.

            :returns: C language unary casting expression.
            :rtype: str

            :raise VisitorError: unexpected number of children.
            '''
            if len(expr_str) not in (1, 2):
                raise VisitorError(
                    f"The C Writer IntrinsicCall cast-style formatter "
                    f"only supports intrinsics with 1 or 2 children, but "
                    f"found '{type_str}' with '{len(expr_str)}' children.")
            # A second child is a Fortran kind: REAL(x, r_def) asks for a
            # particular width. A kind-blind writer cannot honour it, and
            # discarding it is only safe because each cast target here is
            # the widest of its intrinsic, so the result is never narrowed
            # below what was asked for. KokkosWriter overrides this method
            # and casts at the width the region's kind_types give, which is
            # where a caller needing the requested width should look.
            return "(" + type_str + ")" + expr_str[0]

        def cast_function_format(spec, expr_str):
            '''
            :param str spec: the cast target and the function name, joined
                by a colon, as in ``int:round``.
            :param List[str] expr_str: String representation of the operands.

            :returns: C language cast of a unary function expression.
            :rtype: str

            :raise VisitorError: unexpected number of children.
            '''
            if len(expr_str) != 1:
                raise VisitorError(
                    f"The C Writer IntrinsicCall cast-function formatter "
                    f"only supports intrinsics with 1 child, but found "
                    f"'{spec}' with '{len(expr_str)}' children.")
            type_str, function_str = spec.split(":")
            return f"({type_str}){function_str}({expr_str[0]})"

        def fold_format(function_str, expr_str):
            '''
            :param str function_str: Name of the binary function.
            :param List[str] expr_str: String representation of the operands.

            :returns: C language expression folding a variadic Fortran
                intrinsic into nested binary calls, right to left.
            :rtype: str

            :raise VisitorError: unexpected number of children.
            '''
            if len(expr_str) < 2:
                raise VisitorError(
                    f"The C Writer IntrinsicCall fold formatter only "
                    f"supports intrinsics with 2 or more children, but found "
                    f"'{function_str}' with '{len(expr_str)}' children.")
            folded = expr_str[-1]
            for operand in reversed(expr_str[:-1]):
                folded = f"{function_str}({operand}, {folded})"
            return folded

        # Define a map with the intrinsic string and the formatter function
        # associated with each Intrinsic. MAX and MIN are deliberately absent:
        # they are reached only through REAL_INTRINSIC_ALTERNATIVES below, so
        # that an integer MAX raises rather than being written wrongly. C has
        # no standard integer maximum, fmax returns a double, and a
        # conditional expression would evaluate its arguments twice. The
        # refusal is this writer's alone: Kokkos::max and Kokkos::min are
        # type-generic, so KokkosWriter generates both.
        intrinsic_map = {
            IntrinsicCall.Intrinsic.MOD: ("%", binary_operator_format),
            IntrinsicCall.Intrinsic.SIGN: ("copysign", function_format),
            IntrinsicCall.Intrinsic.SIN: ("sin", function_format),
            IntrinsicCall.Intrinsic.COS: ("cos", function_format),
            IntrinsicCall.Intrinsic.TAN: ("tan", function_format),
            IntrinsicCall.Intrinsic.ASIN: ("asin", function_format),
            IntrinsicCall.Intrinsic.ACOS: ("acos", function_format),
            IntrinsicCall.Intrinsic.ATAN: ("atan", function_format),
            IntrinsicCall.Intrinsic.ATAN2: ("atan2", function_format),
            IntrinsicCall.Intrinsic.ABS: ("abs", function_format),
            IntrinsicCall.Intrinsic.EXP: ("exp", function_format),
            IntrinsicCall.Intrinsic.LOG: ("log", function_format),
            IntrinsicCall.Intrinsic.REAL: ("double", cast_format),
            IntrinsicCall.Intrinsic.INT: ("int", cast_format),
            IntrinsicCall.Intrinsic.NINT: ("int:round", cast_function_format),
            IntrinsicCall.Intrinsic.FLOOR: ("int:floor",
                                            cast_function_format),
            IntrinsicCall.Intrinsic.SQRT: ("sqrt", function_format),
            }

        # An intrinsic Fortran overloads on the argument's type is spelt by
        # that type first, so that a real ABS does not reach C's integer
        # ::abs. Everything else, and every argument whose type is not known
        # to be real, falls through to the map; if the intrinsic is not there
        # either, raise an Error.
        alternative = REAL_INTRINSIC_ALTERNATIVES.get(node.intrinsic)
        if alternative and _is_real_argument(node.arguments[0]):
            opstring = alternative
            formatter = (fold_format
                         if node.intrinsic in (IntrinsicCall.Intrinsic.MAX,
                                               IntrinsicCall.Intrinsic.MIN)
                         else function_format)
        else:
            try:
                opstring, formatter = intrinsic_map[node.intrinsic]
            except KeyError as err:
                raise VisitorError(
                    f"The C backend does not support the "
                    f"'{node.intrinsic.name}' intrinsic.") from err

        return formatter(opstring, [self._visit(ch) for ch in node.arguments])


__all__ = ["CIntrinsicsMixin", "REAL_INTRINSIC_ALTERNATIVES"]
