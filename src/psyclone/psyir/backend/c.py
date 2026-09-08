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


'''C PSyIR backend. Generates C code from PSyIR nodes.
Currently limited to just a few PSyIR nodes to support the OpenCL generation,
it needs to be extended for generating pure C code.

'''
from psyclone.psyir.backend.language_writer import LanguageWriter
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import (
    ArrayConstructor, ArrayReference, Assignment, BinaryOperation, Call,
    IntrinsicCall, Literal, Operation, Range, Reference, UnaryOperation)
from psyclone.psyir.symbols import ArrayType, ScalarType


# PSyIR datatypes now support precision as well as intrinsics. It is
# not clear how to map PSyIR intrinsics and precision onto C types,
# see issue #738.
# Mapping from PSyIR types to C data types.
TYPE_MAP_TO_C = {ScalarType.Intrinsic.INTEGER: "int",
                 ScalarType.Intrinsic.CHARACTER: "char",
                 ScalarType.Intrinsic.BOOLEAN: "bool",
                 ScalarType.Intrinsic.REAL: "double"}

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


def _constructor_position(node):
    '''Name the position an array constructor was found in.

    Only ever used to build a refusal, so a position it does not recognise
    is described in general terms rather than being an error of its own.

    :param node: the array constructor whose position is being named.
    :type node: :py:class:`psyclone.psyir.nodes.ArrayConstructor`

    :returns: a phrase naming where the constructor sits.
    :rtype: str

    '''
    parent = node.parent
    if isinstance(parent, ArrayConstructor):
        return "nested inside another array constructor"
    if isinstance(parent, IntrinsicCall):
        return f"as an argument of the '{parent.intrinsic.name}' intrinsic"
    if isinstance(parent, Call):
        return "as an actual argument of a call"
    if isinstance(parent, Operation):
        return "as an operand of an expression"
    return "in an expression"


class CWriter(LanguageWriter):
    '''Implements a PSyIR-to-C back-end for the PSyIR AST.

    :param kwargs: additional keyword arguments provided to the super class.
    :type kwargs: unwrapped dict.

    '''
    def __init__(self, **kwargs):
        # Construct the base class using [] as array parenthesis, and
        # '.' as structure access symbol
        super().__init__(("[", "]"), ".", **kwargs)

    def gen_indices(self, indices, var_name=None):
        '''Given a list of PSyIR nodes representing the dimensions of an
        array, return a list of strings representing those array dimensions.

        :param indices: list of PSyIR nodes.
        :type indices: list of :py:class:`psyclone.psyir.symbols.Node`
        :param str var_name: Name of the field for which the indices are \
            created. The C-interface uses {var_name}LEN{n} as the size \
            of the corresponding dimension `n`.

        :returns: the C representation of the dimensions.
        :rtype: list of str

        '''
        # In C array expressions should be reversed from the PSyIR order
        # (column-major to row-major order) and flattened (1D).

        # This collects the individual terms for each dimension that
        # must be added:
        summands = []
        # This is the ongoing product of all dimension sizes, i.e.
        # ALEN1 * ALEN2 * ...
        multiplicator = ""

        for dimension, child in enumerate(indices):
            expression = self._visit(child)
            dim_str = f"{var_name}LEN{dimension+1}"
            if multiplicator:
                summands.append(expression + " * " + multiplicator)
                multiplicator = multiplicator + " * " + dim_str
            else:
                summands.append(expression)
                multiplicator = dim_str
        # This function must return a list of indices, since in C
        # there is only one dimension, return a one-dimensional list.
        return [" + ".join(summands)]

    def gen_declaration(self, symbol):
        '''
        Generates string representing the C declaration of the symbol. In C
        declarations can be found inside the argument list or with the
        statements, so no indention or punctuation is generated by this method.

        :param symbol: The symbol instance.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: The C declaration of the given symbol.
        :rtype: str

        :raises NotImplementedError: if there are some symbol types or nodes \
            which are not implemented yet.
        '''
        code = ""
        try:
            intrinsic = symbol.datatype.intrinsic
            code = code + TYPE_MAP_TO_C[intrinsic] + " "
        except (AttributeError, KeyError) as err:
            raise NotImplementedError(
                f"Could not generate C definition for variable '{symbol.name}'"
                f", type '{symbol.datatype}' is not yet supported.") from err

        # If the argument is an array, in C language we define it
        # as an unaliased pointer.
        if symbol.is_array:
            code += "* restrict "

        code += symbol.name
        return code

    def gen_local_variable(self, symbol):
        '''
        Generate C code that declares all local symbols in the Symbol Table.

        :param symbol: The symbol instance.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :returns: C language declaration of a local variable.
        :rtype: str
        '''
        return f"{self._nindent}{self.gen_declaration(symbol)};\n"

    def assignment_node(self, node):
        '''This method is called when an Assignment instance is found in the
        PSyIR tree.

        An assignment whose right-hand side is an array constructor is handed
        to :py:meth:`arrayconstructor_node` whole, because C has no value of
        array type and so no right-hand side for this method to write. That
        callback produces the statements the assignment becomes -- one per
        element -- rather than an expression.

        :param node: An Assignment PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.Assignment``

        :returns: The C code as a string.
        :rtype: str

        '''
        if isinstance(node.rhs, ArrayConstructor):
            return self._visit(node.rhs)

        lhs = self._visit(node.lhs)
        rhs = self._visit(node.rhs)

        result = f"{self._nindent}{lhs} = {rhs};\n"
        return result

    @staticmethod
    def _constructor_target(node):
        '''Find where the elements of an array constructor are to be put.

        C has no value of array type, so a constructor can only be written
        where each of its elements already has somewhere to go: as the whole
        right-hand side of an assignment to an array. A braced initialiser is
        not the alternative it looks like, because C accepts one only on a
        declaration and the array assigned to here was declared earlier; and
        anywhere else -- an actual argument, an operand, a nested constructor
        -- the value has to survive as a whole, which needs a temporary array
        this backend does not create.

        The target is named as Fortran subscripts it. The first element goes
        to the declared lower bound of the dimension being filled rather than
        to zero, so that a writer which re-bases subscripts -- as
        :py:class:`psyclone.psyir.backend.kokkos.KokkosWriter` does with a
        View's index offsets -- is handed the same Fortran index here as
        everywhere else and subtracts the origin exactly once.

        :param node: the array constructor being written.
        :type node: :py:class:`psyclone.psyir.nodes.ArrayConstructor`

        :returns: the reference to the target array, the dimension the
            constructor fills, counted from zero, and the Fortran index of
            the first element of that dimension.
        :rtype: Tuple[:py:class:`psyclone.psyir.nodes.Reference`, int, int]

        :raises VisitorError: if the constructor is not the whole right-hand
            side of an assignment, and so would need a temporary array.
        :raises VisitorError: if the target is neither a whole array nor a
            full-extent section of one dimension of an array.
        :raises VisitorError: if that dimension has no literal lower bound in
            its declaration, so the Fortran index of an element is not known.

        '''
        assignment = node.parent
        if (not isinstance(assignment, Assignment)
                or assignment.rhs is not node):
            raise VisitorError(
                f"The C backend cannot write an array constructor "
                f"{_constructor_position(node)}: C has no array-valued "
                f"expression, so this constructor needs a temporary array to "
                f"hold its elements and the backend creates none. Only a "
                f"constructor that is the whole right-hand side of an "
                f"assignment to an array is written, element by element.")

        lhs = assignment.lhs
        dimension = 0
        if isinstance(lhs, ArrayReference):
            sections = [index for index, subscript in enumerate(lhs.indices)
                        if isinstance(subscript, Range)]
            supported = (len(sections) == 1
                         and lhs.is_full_range(sections[0]))
            if supported:
                dimension = sections[0]
        else:
            supported = (isinstance(lhs, Reference) and not lhs.children
                         and isinstance(lhs.symbol.datatype, ArrayType)
                         and len(lhs.symbol.datatype.shape) == 1)
        if not supported:
            target = lhs.name if isinstance(lhs, Reference) else str(lhs)
            raise VisitorError(
                f"The C backend can only write an array constructor into a "
                f"whole rank-1 array or a full-extent section of one "
                f"dimension of an array, but found one assigned to "
                f"'{target}', which is neither: that assignment needs a "
                f"temporary array to hold the constructor.")

        datatype = lhs.symbol.datatype
        bounds = (datatype.shape[dimension]
                  if isinstance(datatype, ArrayType) else None)
        lower = getattr(bounds, "lower", None)
        if not isinstance(lower, Literal):
            raise VisitorError(
                f"The C backend cannot write an array constructor into "
                f"'{lhs.name}' because dimension {dimension + 1} of its "
                f"declaration has no literal lower bound, so the Fortran "
                f"index of each element of the constructor is not known "
                f"here.")
        return lhs, dimension, int(lower.value)

    def arrayconstructor_node(self, node):
        '''This method is called when an ArrayConstructor instance is found
        in the PSyIR tree.

        The constructor is written as one assignment per element rather than
        as a value: ``x = [a, b]`` on an array declared from 1 becomes
        ``x[1] = a; x[2] = b;``. This callback therefore produces whole
        statements, which is why :py:meth:`assignment_node` hands it the
        assignment instead of asking it for a right-hand side, and why every
        other position raises rather than being written. Which positions
        those are, and why, is in :py:meth:`_constructor_target`.

        :param node: an ArrayConstructor PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.ArrayConstructor`

        :returns: The C code as a string.
        :rtype: str

        :raises VisitorError: if the constructor is in a position that would
            need a temporary array; see :py:meth:`_constructor_target`.

        '''
        reference, dimension, first = self._constructor_target(node)

        integer = ScalarType.integer_type()
        result = ""
        for position, element in enumerate(node.children):
            index = Literal(str(first + position), integer)
            if isinstance(reference, ArrayReference):
                indices = [subscript.copy()
                           for subscript in reference.indices]
                indices[dimension] = index
            else:
                indices = [index]
            target = ArrayReference.create(reference.symbol, indices)
            result += (f"{self._nindent}{self._visit(target)} = "
                       f"{self._visit(element)};\n")
        return result

    def literal_node(self, node):
        '''This method is called when a Literal instance is found in the PSyIR
        tree.

        :param node: A Literal PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.Literal`

        :returns: The C code as a string.
        :rtype: str

        '''
        if node.datatype.intrinsic == ScalarType.Intrinsic.REAL:
            try:
                _ = int(node.value)
                result = node.value + ".0"
            except ValueError:
                # It is already formatted as a real.
                result = node.value
        else:
            result = node.value
        return result

    def ifblock_node(self, node):
        '''This method is called when an IfBlock instance is found in the
        PSyIR tree.

        :param node: An IfBlock PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.IfBlock`

        :returns: The C code as a string.
        :rtype: str

        :raises VisitorError: If node has fewer children than expected.

        '''
        if len(node.children) < 2:
            raise VisitorError(
                f"IfBlock malformed or incomplete. It should have at least "
                f"2 children, but found {len(node.children)}.")

        condition = self._visit(node.condition)

        self._depth += 1
        if_body = ""
        for child in node.if_body:
            if_body += self._visit(child)
        else_body = ""
        # node.else_body is None if there is no else clause.
        if node.else_body:
            for child in node.else_body:
                else_body += self._visit(child)
        self._depth -= 1

        if else_body:
            result = (
                f"{self._nindent}if ({condition}) {{\n"
                f"{if_body}"
                f"{self._nindent}}} else {{\n"
                f"{else_body}"
                f"{self._nindent}}}\n")
        else:
            result = (
                f"{self._nindent}if ({condition}) {{\n"
                f"{if_body}"
                f"{self._nindent}}}\n")
        return result

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
            BinaryOperation.Operator.POW: ("pow", function_format),
            BinaryOperation.Operator.EQ: ("==", operator_format),
            BinaryOperation.Operator.NE: ("!=", operator_format),
            BinaryOperation.Operator.LT: ("<", operator_format),
            BinaryOperation.Operator.LE: ("<=", operator_format),
            BinaryOperation.Operator.GT: (">", operator_format),
            BinaryOperation.Operator.GE: (">=", operator_format),
            BinaryOperation.Operator.AND: ("&&", operator_format),
            BinaryOperation.Operator.OR: ("||", operator_format),
            }

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

    def exit_node(self, _):
        '''This method is called when an Exit instance is found in the
        PSyIR tree.

        Fortran's unlabelled EXIT and C's break both leave the innermost
        enclosing loop, so the translation needs nothing else. The node
        itself checks that there is a loop to leave.

        :param node: an Exit PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.Exit`

        :returns: the C code as a string.
        :rtype: str

        '''
        return f"{self._nindent}break;\n"

    def return_node(self, _):
        '''This method is called when a Return instance is found in
        the PSyIR tree.

        :param node: A Return PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.Return`

        :returns: The C code as a string.
        :rtype: str

        '''
        return f"{self._nindent}return;\n"

    def codeblock_node(self, _):
        '''This method is called when a CodeBlock instance is found in the
        PSyIR tree. At the moment all CodeBlocks contain Fortran fparser
        code.

        :raises VisitorError: The CodeBlock can not be translated to C.

        '''
        raise VisitorError("CodeBlocks can not be translated to C.")

    @staticmethod
    def _loop_counts_down(step_expr):
        '''Whether a loop with this step expression is known to count down.

        Fortran's DO runs "while still in range" and needs no direction in
        its text, but C tests one way or the other, so the direction has to
        be decided here. It can only be decided for a step whose sign is
        visible in the tree; a step that is a runtime value is assumed to be
        positive, as it was before this returned anything.

        :param step_expr: the loop's step expression.
        :type step_expr: :py:class:`psyclone.psyir.nodes.Node`

        :returns: whether the step is a negative literal.
        :rtype: bool

        '''
        if isinstance(step_expr, Literal):
            return step_expr.value.startswith("-")
        if (isinstance(step_expr, UnaryOperation)
                and step_expr.operator == UnaryOperation.Operator.MINUS
                and isinstance(step_expr.children[0], Literal)):
            return not step_expr.children[0].value.startswith("-")
        return False

    def loop_node(self, node):
        '''This method is called when a Loop instance is found in the
        PSyIR tree.

        The loop's continuation test follows the sign of its step, so that a
        Fortran countdown such as ``do k = n, 1, -1`` becomes ``for(k=n;
        k>=1; k+=-1)`` rather than a loop whose body never runs. Only a step
        whose sign is visible in the tree is followed; see
        :py:meth:`_loop_counts_down`.

        :param node: a Loop PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.Loop`

        :returns: the loop node converted into a (language specific) string.
        :rtype: str

        '''
        start = self._visit(node.start_expr)
        stop = self._visit(node.stop_expr)
        step = self._visit(node.step_expr)
        variable_name = node.variable.name
        test = ">=" if self._loop_counts_down(node.step_expr) else "<="

        self._depth += 1
        body = ""
        for child in node.loop_body:
            body += self._visit(child)
        self._depth -= 1

        return f"{self._nindent}for({variable_name}={start}; "\
               f"{variable_name}{test}{stop}; {variable_name}+={step})\n"\
               f"{self._nindent}{{\n{body}{self._nindent}}}\n"

    def whileloop_node(self, node):
        '''This method is called when a WhileLoop instance is found in the
        PSyIR tree.

        Fortran's ``DO WHILE`` and C's ``while`` test the same condition in
        the same place, so the translation is the text and nothing else. In
        particular there is no counterpart here to the step-direction
        reasoning :py:meth:`loop_node` has to do: a while loop states its own
        continuation test rather than leaving it to be inferred.

        :param node: a WhileLoop PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.WhileLoop`

        :returns: the C code as a string.
        :rtype: str

        '''
        condition = self._visit(node.condition)

        self._depth += 1
        body = ""
        for child in node.loop_body:
            body += self._visit(child)
        self._depth -= 1

        return (f"{self._nindent}while ({condition}) {{\n"
                f"{body}"
                f"{self._nindent}}}\n")

    def regiondirective_node(self, node):
        '''This method is called when an RegionDirective instance is found in
        the PSyIR tree. It returns the opening and closing directives, and
        the statements in between as a string.

        :param node: a RegionDirective PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.RegionDirective`

        :returns: the C code as a string.
        :rtype: str

        '''
        # Note that {{ is replaced with a single { in the format call
        result_list = [f"{self._nindent}#pragma {node.begin_string()}"]

        clause_list = []
        for clause in node.clauses:
            val = self._visit(clause)
            # Some clauses return empty strings if they should not
            # generate any output (e.g. private clause with no children).
            if val != "":
                clause_list.append(val)
        # Add a space only if there are clauses
        if len(clause_list) > 0:
            result_list.append(" ")
        result_list.append(", ".join(clause_list))
        result_list.append(f"\n{self._nindent}{{\n")

        self._depth += 1
        for child in node.dir_body:
            result_list.append(self._visit(child))
        self._depth -= 1
        # Note that }} is replaced with a single } in the format call
        result_list.append(f"{self._nindent}}}\n")
        return "".join(result_list)

    def standalonedirective_node(self, node):
        '''This method is called when an StandaloneDirective instance is
        found in the PSyIR tree. It returns the opening and closing directives,
        and the statements in between as a string.

        :param node: a StandaloneDirective PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.StandaloneDirective`

        :returns: the C code as a string.
        :rtype: str

        '''
        result_list = [f"{self._nindent}#pragma {node.begin_string()}\n"]
        return "".join(result_list)

    def filecontainer_node(self, node):
        '''This method is called when a FileContainer instance is found in
        the PSyIR tree.

        :param node: a Container PSyIR node.
        :type node: :py:class:`psyclone.psyir.nodes.FileContainer`

        :returns: the C code.
        :rtype: str

        '''
        result = ""
        for child in node.children:
            result += self._visit(child)
        return result
