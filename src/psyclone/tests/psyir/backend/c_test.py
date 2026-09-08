# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2019-2026, Science and Technology Facilities Council.
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
# Author S. Siso, STFC Daresbury Lab
# Modified by A. R. Porter and R. W. Ford, STFC Daresbury Lab
# -----------------------------------------------------------------------------

'''Performs pytest tests on the psyclone.psyir.backend.c module'''

import pytest

from psyclone.errors import GenerationError, InternalError
from psyclone.psyir.backend.c import CWriter, _is_real_argument
from psyclone.psyir.backend.visitor import VisitorError
from psyclone.psyir.nodes import (
    ArrayConstructor, ArrayReference, Assignment, BinaryOperation, Call,
    CodeBlock, Exit, IfBlock, Literal, Node, Reference, Return, Schedule,
    UnaryOperation, Loop, OMPTaskloopDirective, OMPMasterDirective,
    OMPParallelDirective, IntrinsicCall, OMPBarrierDirective)
from psyclone.psyir.symbols import (
    ArgumentInterface, ArrayType, ScalarType, DataSymbol, RoutineSymbol,
    UnresolvedType)


def test_cw_gen_declaration():
    '''Check the CWriter class gen_declaration method produces
    the expected declarations.

    '''
    cwriter = CWriter()

    # Basic entries
    symbol = DataSymbol("dummy1", ScalarType.integer_type())
    result = cwriter.gen_declaration(symbol)
    assert result == "int dummy1"

    symbol = DataSymbol("dummy1", ScalarType.character_type())
    result = cwriter.gen_declaration(symbol)
    assert result == "char dummy1"

    symbol = DataSymbol("dummy1", ScalarType.boolean_type())
    result = cwriter.gen_declaration(symbol)
    assert result == "bool dummy1"

    # Array argument
    array_type = ArrayType(ScalarType.real_type(),
                           [ArrayType.Extent.ATTRIBUTE,
                            ArrayType.Extent.ATTRIBUTE,
                            ArrayType.Extent.ATTRIBUTE])
    symbol = DataSymbol("dummy2", array_type,
                        interface=ArgumentInterface(
                            ArgumentInterface.Access.READ))
    result = cwriter.gen_declaration(symbol)
    assert result == "double * restrict dummy2"

    # Array with unknown access
    array_type = ArrayType(ScalarType.integer_type(), [2, 4, 2])
    symbol = DataSymbol("dummy2", array_type,
                        interface=ArgumentInterface(
                            ArgumentInterface.Access.UNKNOWN))
    result = cwriter.gen_declaration(symbol)
    assert result == "int * restrict dummy2"

    # Check invalid datatype produces and error
    symbol._datatype = "invalid"
    with pytest.raises(NotImplementedError) as error:
        _ = cwriter.gen_declaration(symbol)
    assert "Could not generate C definition for variable 'dummy2', " \
        "type 'invalid' is not yet supported." in str(error.value)


def test_cw_gen_local_variable(monkeypatch):
    '''Check the CWriter class gen_local_variable method produces
    the expected declarations.

    '''
    cwriter = CWriter()

    monkeypatch.setattr(cwriter, "gen_declaration",
                        lambda x: "<declaration>")

    # Local variables are declared as single statements
    symbol = DataSymbol("dummy1", ScalarType.integer_type())
    result = cwriter.gen_local_variable(symbol)
    # Result should include the mocked gen_declaration and ';\n'
    assert result == "<declaration>;\n"


def test_cw_exception():
    '''Check the CWriter class instance raises an exception if an
    unsupported PSyIR node is found.

    '''
    # Define a Node which will be unsupported by the visitor
    class Unsupported(Node):
        '''A PSyIR node that will not be supported by the C visitor.'''
    # pylint: enable=abstract-method

    unsupported = Unsupported()

    cwriter = CWriter()
    with pytest.raises(VisitorError) as excinfo:
        _ = cwriter(unsupported)
    assert "Unsupported node 'Unsupported' found" in str(excinfo.value)


def test_cw_literal():
    '''Check the CWriter class literal method correctly prints
    out the C representation of a Literal.

    '''

    cwriter = CWriter()

    lit = Literal('1', ScalarType.integer_type())
    assert cwriter(lit) == '1'

    lit = Literal('1', ScalarType.real_type())
    assert cwriter(lit) == '1.0'

    # Test that scientific notation is output correctly
    lit = Literal("3e5", ScalarType.real_type(), None)
    assert cwriter(lit) == '3e5'


def test_cw_assignment():
    '''Check the CWriter class assignment method generate the appropriate
    output.

    '''
    assignment = Assignment.create(
        Reference(DataSymbol('a', ScalarType.real_type())),
        Reference(DataSymbol('b', ScalarType.real_type())))
    # Generate C from the PSyIR schedule
    cwriter = CWriter()
    result = cwriter(assignment)
    assert result == "a = b;\n"


def test_cw_array():
    '''Check the CWriter class array method correctly prints
    out the C representation of an array.

    '''
    cwriter = CWriter()

    symbol = DataSymbol('a', ScalarType.real_type())
    arr = ArrayReference(symbol)
    lit = Literal('0.0', ScalarType.real_type())
    assignment = Assignment.create(arr, lit)

    # An array without any children (dimensions) should produce an error.
    with pytest.raises(VisitorError) as excinfo:
        result = cwriter(assignment)
    assert "Incomplete ArrayReference node (for symbol 'a') found: " \
           "must have one or more children." in str(excinfo.value)

    # Dimensions can be references, literals or operations
    arr.addchild(Reference(DataSymbol('b', ScalarType.integer_type())))
    arr.addchild(Literal('1', ScalarType.integer_type()))
    uop = UnaryOperation.create(UnaryOperation.Operator.MINUS,
                                Literal('2', ScalarType.integer_type()))
    arr.addchild(uop)

    result = cwriter(assignment)
    # Results is reversed and flatten (row-major 1D), so
    # a[b, 1, -2] becomes a[(-2) * aLEN2 * aLEN1 + 1 * aLEN1 + b]
    # dimensions are called <name>LEN<dimension> by convention
    assert result == "a[b + 1 * aLEN1 + (-2) * aLEN1 * aLEN2] = 0.0;\n"


def test_cw_ifblock():
    '''Check the CWriter class ifblock method correctly prints out the
    C representation.

    '''

    # Try with just an IfBlock node
    ifblock = IfBlock()
    cwriter = CWriter()
    with pytest.raises(VisitorError) as err:
        _ = cwriter(ifblock)
    assert ("IfBlock malformed or incomplete. It should have "
            "at least 2 children, but found 0." in str(err.value))

    # Add the if condition
    ifblock.addchild(Reference(DataSymbol('a', ScalarType.real_type())))
    with pytest.raises(VisitorError) as err:
        _ = cwriter(ifblock)
    assert ("IfBlock malformed or incomplete. It should have "
            "at least 2 children, but found 1." in str(err.value))

    # Fill the if_body
    ifblock.addchild(Schedule(parent=ifblock))
    ifblock.if_body.addchild(Return(parent=ifblock.if_body))
    if_only = cwriter(ifblock)
    assert if_only == (
        "if (a) {\n"
        "  return;\n"
        "}\n")
    # Fill the else_body
    ifblock.addchild(Schedule(parent=ifblock))

    condition = Reference(DataSymbol('b', ScalarType.real_type()))
    then_content = [Return()]
    else_content = [Return()]
    ifblock2 = IfBlock.create(condition, then_content, else_content)
    ifblock.else_body.addchild(ifblock2)

    result = cwriter(ifblock)
    assert result == (
        "if (a) {\n"
        "  return;\n"
        "} else {\n"
        "  if (b) {\n"
        "    return;\n"
        "  } else {\n"
        "    return;\n"
        "  }\n"
        "}\n")


def test_cw_return():
    '''Check the CWriter class return method correctly prints out the
    C representation.

    '''
    cwriter = CWriter()
    result = cwriter(Return())
    assert "return;\n" in result


def test_cw_exit(fortran_reader):
    '''Check the CWriter class writes an Exit as C's break, in a counted
    loop and in a while loop alike.

    Fortran's unlabelled EXIT and C's break both leave the innermost
    enclosing loop, so no other statement is needed to carry it across.

    '''
    code = '''
        module test
        contains
        subroutine tmp(a)
          integer :: i, a
          do i = 1, 20
            if (a > i) then
              exit
            end if
            a = a + i
          enddo
          do while (a < 100)
            a = a + 1
            if (a == 50) exit
          end do
        end subroutine tmp
        end module test'''
    container = fortran_reader.psyir_from_source(code).children[0]
    module = container.children[0]

    cwriter = CWriter()
    counted = cwriter(module[0])
    assert ("for(i=1; i<=20; i+=1)\n{\n  if ((a > i)) {\n    break;\n  }"
            in counted)
    assert "break;" in cwriter(module[1])


def test_cw_exit_needs_a_loop():
    '''Check that the CWriter refuses an Exit with no loop to leave: a
    stray break would either not compile or leave a switch instead.

    '''
    cwriter = CWriter()
    with pytest.raises(GenerationError) as error:
        _ = cwriter(Exit())
    assert "Exit must be inside a loop" in str(error.value)


def test_cw_codeblock():
    '''Check the CWriter class codeblock method raises the expected
    error.
    '''

    cblock = CodeBlock([], "dummy")
    cwriter = CWriter()

    with pytest.raises(VisitorError) as error:
        _ = cwriter(cblock)
    assert "CodeBlocks can not be translated to C." in str(error.value)


def test_cw_unaryoperator():
    '''Check the CWriter class unary_operation method correctly prints out
    the C representation of any given UnaryOperation.

    '''
    cwriter = CWriter()

    # Test UnaryOperation without children.
    unary_operation = UnaryOperation(UnaryOperation.Operator.MINUS)
    with pytest.raises(VisitorError) as err:
        _ = cwriter(unary_operation)
    assert ("UnaryOperation malformed or incomplete. It should have "
            "exactly 1 child, but found 0." in str(err.value))

    # Add child
    ref1 = Literal("a", ScalarType.character_type(), unary_operation)
    unary_operation.addchild(ref1)
    assert cwriter(unary_operation) == '(-a)'

    # Test all supported Operators
    test_list = ((UnaryOperation.Operator.PLUS, '(+a)'),
                 (UnaryOperation.Operator.MINUS, '(-a)'),
                 (UnaryOperation.Operator.NOT, '(!a)'))

    for operator, expected in test_list:
        unary_operation._operator = operator
        assert cwriter(unary_operation) in expected

    # Test that an unsupported operator raises an error
    class Unsupported():
        ''' Mock Unsupported object '''

    unary_operation._operator = Unsupported
    with pytest.raises(NotImplementedError) as err:
        _ = cwriter(unary_operation)
    assert "The C backend does not support the '" in str(err.value)
    assert "' operator." in str(err.value)


def test_cw_binaryoperator():
    '''Check the CWriter class binary_operation method correctly
    prints out the C representation of any given BinaryOperation.

    '''
    cwriter = CWriter()

    # Test UnaryOperation without children.
    binary_operation = BinaryOperation(BinaryOperation.Operator.ADD)
    with pytest.raises(VisitorError) as err:
        _ = cwriter(binary_operation)
    assert ("BinaryOperation malformed or incomplete. It should have "
            "exactly 2 children, but found 0." in str(err.value))

    # Test with children
    ref1 = Reference(DataSymbol("a", ScalarType.real_type()))
    ref2 = Reference(DataSymbol("b", ScalarType.real_type()))
    binary_operation = BinaryOperation.create(BinaryOperation.Operator.ADD,
                                              ref1, ref2)
    assert cwriter(binary_operation) == '(a + b)'

    # Test all supported Operators
    test_list = ((BinaryOperation.Operator.ADD, '(a + b)'),
                 (BinaryOperation.Operator.SUB, '(a - b)'),
                 (BinaryOperation.Operator.MUL, '(a * b)'),
                 (BinaryOperation.Operator.DIV, '(a / b)'),
                 (BinaryOperation.Operator.POW, 'pow(a, b)'),
                 (BinaryOperation.Operator.EQ, '(a == b)'),
                 (BinaryOperation.Operator.NE, '(a != b)'),
                 (BinaryOperation.Operator.GT, '(a > b)'),
                 (BinaryOperation.Operator.GE, '(a >= b)'),
                 (BinaryOperation.Operator.LT, '(a < b)'),
                 (BinaryOperation.Operator.LE, '(a <= b)'),
                 (BinaryOperation.Operator.OR, '(a || b)'),
                 (BinaryOperation.Operator.AND, '(a && b)'))

    for operator, expected in test_list:
        binary_operation._operator = operator
        assert cwriter(binary_operation) == expected

    # Test that an unsupported operator raises a error
    class Unsupported():
        '''Dummy class'''

    binary_operation._operator = Unsupported
    with pytest.raises(VisitorError) as err:
        _ = cwriter(binary_operation)
    assert "The C backend does not support the '" in str(err.value)
    assert "' operator." in str(err.value)


def test_cw_intrinsiccall():
    '''Check the CWriter class intrinsiccall method correctly prints out
    the C representation of any given Intrinsic.

    '''
    cwriter = CWriter()

    # Test all supported Intrinsics with 1 argument
    test_list = ((IntrinsicCall.Intrinsic.SQRT, 'sqrt(a)'),
                 (IntrinsicCall.Intrinsic.COS, 'cos(a)'),
                 (IntrinsicCall.Intrinsic.SIN, 'sin(a)'),
                 (IntrinsicCall.Intrinsic.TAN, 'tan(a)'),
                 (IntrinsicCall.Intrinsic.ACOS, 'acos(a)'),
                 (IntrinsicCall.Intrinsic.ASIN, 'asin(a)'),
                 (IntrinsicCall.Intrinsic.ATAN, 'atan(a)'),
                 (IntrinsicCall.Intrinsic.ABS, 'fabs(a)'),
                 (IntrinsicCall.Intrinsic.EXP, 'exp(a)'),
                 (IntrinsicCall.Intrinsic.LOG, 'log(a)'),
                 (IntrinsicCall.Intrinsic.NINT, '(int)round(a)'),
                 (IntrinsicCall.Intrinsic.FLOOR, '(int)floor(a)'),
                 (IntrinsicCall.Intrinsic.REAL, '(double)a'))
    ref1 = Reference(DataSymbol("a", ScalarType.real_type()))
    for intrinsic, expected in test_list:
        icall = IntrinsicCall.create(intrinsic, [ref1.copy()])
        assert cwriter(icall) == expected

    # Check that operator-style formatting with a number of children different
    # than 2 produces an error. The argument has to be an integer: a real MOD
    # is now written as fmod, so '%' is only reached on the integer path.
    with pytest.raises(VisitorError) as err:
        icall = IntrinsicCall(IntrinsicCall.Intrinsic.MOD)
        icall.addchild(Reference(DataSymbol("i", ScalarType.integer_type())))
        _ = cwriter(icall)
    assert ("The C Writer binary_operator formatter for IntrinsicCall only "
            "supports intrinsics with 2 children, but found '%' with '1' "
            "children." in str(err.value))

    # Test all supported Intrinsics with 2 arguments
    test_list = (
                 (IntrinsicCall.Intrinsic.MOD, 'fmod(a, b)'),
                 (IntrinsicCall.Intrinsic.SIGN, 'copysign(a, b)'),
                 (IntrinsicCall.Intrinsic.ATAN2, 'atan2(a, b)'),
                 (IntrinsicCall.Intrinsic.MIN, 'fmin(a, b)'),
    )
    ref1 = Reference(DataSymbol("a", ScalarType.real_type()))
    ref2 = Reference(DataSymbol("b", ScalarType.real_type()))
    for intrinsic, expected in test_list:
        icall = IntrinsicCall.create(intrinsic, [ref1.copy(), ref2.copy()])
        assert cwriter(icall) == expected

    # A variadic Fortran intrinsic folds right to left into nested binary
    # C calls, since fmax and fmin take exactly two arguments.
    ref3 = Reference(DataSymbol("c", ScalarType.real_type()))
    icall = IntrinsicCall.create(IntrinsicCall.Intrinsic.MAX,
                                 [ref1.copy(), ref2.copy(), ref3.copy()])
    assert cwriter(icall) == 'fmax(a, fmax(b, c))'

    # A cast now accepts the Fortran kind as a second child, and discards
    # it: REAL(a, r_def) asks for a width a kind-blind writer cannot honour.
    icall = IntrinsicCall(IntrinsicCall.Intrinsic.REAL)
    icall.addchild(ref1.copy())
    icall.addchild(ref2.copy())
    assert cwriter(icall) == '(double)a'

    # Three children is still an error.
    with pytest.raises(VisitorError) as err:
        icall = IntrinsicCall(IntrinsicCall.Intrinsic.REAL)
        for _ in range(3):
            icall.addchild(ref1.copy())
        _ = cwriter(icall)
    assert ("The C Writer IntrinsicCall cast-style formatter only supports "
            "intrinsics with 1 or 2 children, but found 'double' with '3' "
            "children." in str(err.value))

    # The cast-function formatter takes exactly one child.
    with pytest.raises(VisitorError) as err:
        icall = IntrinsicCall(IntrinsicCall.Intrinsic.NINT)
        icall.addchild(ref1.copy())
        icall.addchild(ref2.copy())
        _ = cwriter(icall)
    assert ("The C Writer IntrinsicCall cast-function formatter only supports "
            "intrinsics with 1 child, but found 'int:round' with '2' children."
            in str(err.value))

    # The fold formatter takes at least two.
    with pytest.raises(VisitorError) as err:
        icall = IntrinsicCall(IntrinsicCall.Intrinsic.MAX)
        icall.addchild(ref1.copy())
        _ = cwriter(icall)
    assert ("The C Writer IntrinsicCall fold formatter only supports "
            "intrinsics with 2 or more children, but found 'fmax' with '1' "
            "children." in str(err.value))


def test_cw_intrinsiccall_integer():
    '''Check that the intrinsics Fortran overloads on the argument's type
    are written by that type: an integer keeps C's integer spelling where a
    real takes the maths-library one, and an integer maximum, which C has no
    standard spelling for, is refused rather than written wrongly.

    '''
    cwriter = CWriter()
    int1 = Reference(DataSymbol("i", ScalarType.integer_type()))
    int2 = Reference(DataSymbol("j", ScalarType.integer_type()))

    icall = IntrinsicCall.create(IntrinsicCall.Intrinsic.ABS, [int1.copy()])
    assert cwriter(icall) == 'abs(i)'

    icall = IntrinsicCall.create(IntrinsicCall.Intrinsic.MOD,
                                 [int1.copy(), int2.copy()])
    assert cwriter(icall) == '(i % j)'

    # fmax returns a double, C has no standard integer maximum, and a
    # conditional expression would evaluate its arguments twice. KokkosWriter
    # handles both types, with the type-generic Kokkos::max.
    for intrinsic in (IntrinsicCall.Intrinsic.MAX,
                      IntrinsicCall.Intrinsic.MIN):
        with pytest.raises(VisitorError) as err:
            icall = IntrinsicCall.create(intrinsic,
                                         [int1.copy(), int2.copy()])
            _ = cwriter(icall)
        assert (f"The C backend does not support the '{intrinsic.name}' "
                f"intrinsic." in str(err.value))


def test_cw_is_real_argument():
    '''Check that _is_real_argument answers "no" rather than raising for
    every reason it might not know the type, since the kind-blind default
    path has to stay reachable for a caller probing the writer with
    synthetic arguments.

    '''
    assert _is_real_argument(
        Reference(DataSymbol("a", ScalarType.real_type()))) is True
    assert _is_real_argument(
        Reference(DataSymbol("i", ScalarType.integer_type()))) is False
    assert _is_real_argument(
        Reference(DataSymbol("u", UnresolvedType()))) is False
    # An incomplete tree: ArrayReference.datatype raises rather than
    # returning anything for a reference carrying no indices.
    malformed = ArrayReference(
        DataSymbol("x", ArrayType(ScalarType.real_type(), [10])))
    with pytest.raises(InternalError):
        _ = malformed.datatype
    assert _is_real_argument(malformed) is False


def test_cw_loop(fortran_reader):
    '''Tests writing out a Loop node in C. It parses Fortran code
    and outputs it as C. Note that this is atm a literal translation,
    the loops are not functionally identical to Fortran, see TODO #523.

    '''
    # Generate PSyIR from Fortran code.
    code = '''
        module test
        contains
        subroutine tmp(b)
          integer :: i, a
          integer, dimension(:) :: b
          do i = 1, 20, 2
            a = 2 * i
          enddo
        end subroutine tmp
        end module test'''
    container = fortran_reader.psyir_from_source(code).children[0]
    module = container.children[0]

    cwriter = CWriter()
    result = cwriter(module[0])
    correct = '''for(i=1; i<=20; i+=2)
{
  a = (2 * i);
}'''
    result = cwriter(module[0])
    assert correct in result


def test_cw_loop_counts_down(fortran_reader):
    '''Tests that a Fortran countdown keeps running in C.

    Fortran's DO runs while the variable is still in range and so needs no
    direction in its text, but C tests one way or the other. A negative step
    written with C's ascending test compiles, links and runs zero iterations
    -- a silent wrong answer rather than a diagnosable one.

    '''
    code = '''
        module test
        contains
        subroutine tmp(b)
          integer :: i, n
          integer, dimension(:) :: b
          do i = n, 1, -1
            b(i) = i
          enddo
          do i = n, 1, -2
            b(i) = i
          enddo
        end subroutine tmp
        end module test'''
    container = fortran_reader.psyir_from_source(code).children[0]
    module = container.children[0]

    cwriter = CWriter()
    assert 'for(i=n; i>=1; i+=(-1))' in cwriter(module[0])
    assert 'for(i=n; i>=1; i+=(-2))' in cwriter(module[1])


def test_cw_loop_step_of_unknown_sign(fortran_reader):
    '''Tests that a step whose sign is not in the tree is taken as ascending.

    The direction can only be followed where it is visible. A step that is a
    runtime value keeps the ascending test it has always had, rather than the
    writer refusing a loop it used to generate.

    '''
    code = '''
        module test
        contains
        subroutine tmp(b, s)
          integer :: i, n
          integer, intent(in) :: s
          integer, dimension(:) :: b
          do i = 1, n, s
            b(i) = i
          enddo
        end subroutine tmp
        end module test'''
    container = fortran_reader.psyir_from_source(code).children[0]
    module = container.children[0]

    assert 'for(i=1; i<=n; i+=s)' in CWriter()(module[0])


def test_cw_unsupported_intrinsiccall():
    ''' Check the CWriter class SIZE intrinsic raises the expected error since
    there is no C equivalent. '''
    cwriter = CWriter()
    arr = ArrayReference(DataSymbol('a', ScalarType.integer_type()))
    lit = Literal('1', ScalarType.integer_type())
    size = IntrinsicCall.create(IntrinsicCall.Intrinsic.SIZE,
                                [arr, ("dim", lit)])
    lhs = Reference(DataSymbol('length', ScalarType.integer_type()))
    assignment = Assignment.create(lhs, size)

    with pytest.raises(VisitorError) as excinfo:
        cwriter(assignment)
    assert ("The C backend does not support the 'SIZE' intrinsic."
            in str(excinfo.value))


def test_cw_structureref(fortran_reader):
    ''' Test the CWriter support for StructureReference. '''
    code = '''
        module test
        contains
        subroutine tmp()
          type :: my_type
            integer                   :: b
            integer, dimension(10,10) :: c
            integer, dimension(10)    :: d
          end type my_type
          type(my_type) :: a, b(5)
          integer :: i
          a%b = a%c(1,2) + b(3)%d(i)%e(i+1) + a%f%g(3)
        end subroutine tmp
        end module test'''
    container = fortran_reader.psyir_from_source(code).children[0]
    module = container.children[0]

    cwriter = CWriter()
    result = cwriter(module[0])
    correct = "a.b = ((a.c[1 + 2 * cLEN1] + b[3].d[i].e[(i + 1)]) + a.f.g[3])"
    assert correct in result

    module[0].children[0]._children = []
    with pytest.raises(VisitorError) as err:
        _ = cwriter(module[0])
    assert "A StructureReference must have a single child but the " \
           "reference to symbol 'a' has 0." in str(err.value)

    ref = module[0].children[0]
    ref._children = [Literal("1", ScalarType.integer_type())]
    with pytest.raises(VisitorError) as err:
        # We can't call cwriter(), it will complain about having a Literal
        # node which is invalid. So call _visit()
        _ = cwriter._visit(ref)
    assert "A StructureReference must have a single child which is a " \
           "sub-class of Member but the reference to symbol 'a' has a " \
           "child of type " in str(err.value)


def test_cw_arraystructureref(fortran_reader):
    ''' Test the CWriter support for ArrayStructureReference. '''
    code = '''
        module test
        contains
        subroutine tmp()
          type :: my_type
            integer                   :: b
            integer, dimension(10,10) :: c
            integer, dimension(10)    :: d
          end type my_type
          type(my_type) :: a, b(5)
          integer :: i
          b(5)%d(1) = 1
        end subroutine tmp
        end module test'''
    container = fortran_reader.psyir_from_source(code).children[0]
    module = container.children[0]

    cwriter = CWriter()
    result = cwriter(module[0])
    correct = "b[5].d[1] = 1"
    assert correct in result

    array_ref = module[0].children[0]
    array_ref._children = []
    with pytest.raises(VisitorError) as err:
        # We can't call cwriter(), it will complain about having a Literal
        # node which is invalid. So call _visit()
        _ = cwriter._visit(array_ref)
    assert "An ArrayOfStructuresReference must have at least two children " \
           "but found 0" in str(err.value)

    array_ref._children = [Literal("1", ScalarType.integer_type()),
                           Literal("1", ScalarType.integer_type())]
    with pytest.raises(VisitorError) as err:
        # We can't call cwriter(), it will complain about having a Literal
        # node which is invalid. So call _visit()
        _ = cwriter._visit(array_ref)
    assert "An ArrayOfStructuresReference must have a Member as its first " \
           "child but found 'Literal'" in str(err.value)


def test_cw_directive_with_clause(fortran_reader):
    '''Test that a PSyIR directive with clauses is translated to
    the required C code.

    '''
    cwriter = CWriter()
    # Generate PSyIR from Fortran code.
    code = (
        "program test\n"
        "  integer, parameter :: n=20\n"
        "  integer :: i\n"
        "  real :: a(n)\n"
        "  do i=1,n\n"
        "    a(i) = 0.0\n"
        "  end do\n"
        "end program test")
    container = fortran_reader.psyir_from_source(code)
    schedule = container.children[0]
    loops = schedule.walk(Loop)
    loop = loops[0].detach()
    directive = OMPTaskloopDirective(children=[loop], num_tasks=32,
                                     nogroup=True)
    master = OMPMasterDirective(children=[directive])
    parallel = OMPParallelDirective.create(children=[master])
    schedule.addchild(parallel, 0)
    # Add a barrier to cover the StandaloneDirective visitor
    parallel.children[0].addchild(OMPBarrierDirective(), 0)

    assert '''\
#pragma omp parallel default(shared), private(i)
{
  #pragma omp barrier
  #pragma omp master
  {
    #pragma omp taskloop num_tasks(32), nogroup
    {
      for(i=1; i<=n; i+=1)
      {
        a[i] = 0.0;
      }
    }
  }
}
''' == cwriter(schedule.children[0])


def test_cw_while_loop(fortran_reader):
    '''Tests writing out a WhileLoop node in C.

    Fortran's ``DO WHILE`` and C's ``while`` say the same thing, so the only
    things this can get wrong are the punctuation and the indentation of a
    body that is more than one statement deep.

    '''
    code = '''
        module test
        contains
        subroutine tmp(n)
          integer, intent(in) :: n
          integer :: i
          i = 0
          do while (i < n)
            i = i + 1
            if (i > 3) then
              i = i + 2
            end if
          enddo
        end subroutine tmp
        end module test'''
    module = fortran_reader.psyir_from_source(code).children[0].children[0]

    assert CWriter()(module[1]) == (
        "while ((i < n)) {\n"
        "  i = (i + 1);\n"
        "  if ((i > 3)) {\n"
        "    i = (i + 2);\n"
        "  }\n"
        "}\n")


def test_cw_array_constructor_of_literals(fortran_reader):
    '''Tests that a constructor assigned to a whole array is written out as
    one assignment per element.

    A braced initialiser would be the obvious translation and is not a legal
    one: C accepts a braced list only on a declaration, and the array being
    assigned to here was declared earlier in the body. Element assignments
    are what is left.

    The elements are placed from the array's declared lower bound rather than
    from zero, so ``y`` below starts at 0 and ``x`` at 1. The Fortran index
    is what a writer that re-bases subscripts expects to be given.

    '''
    code = '''
        module test
        contains
        subroutine tmp()
          integer :: x(4)
          real :: y(0:2)
          x = [1, 2, 3, 4]
          y = (/ 1.0, 2.0, 3.0 /)
        end subroutine tmp
        end module test'''
    module = fortran_reader.psyir_from_source(code).children[0].children[0]
    cwriter = CWriter()

    assert cwriter(module[0]) == (
        "x[1] = 1;\n"
        "x[2] = 2;\n"
        "x[3] = 3;\n"
        "x[4] = 4;\n")
    assert cwriter(module[1]) == (
        "y[0] = 1.0;\n"
        "y[1] = 2.0;\n"
        "y[2] = 3.0;\n")


def test_cw_array_constructor_into_a_section(fortran_reader):
    '''Tests that a constructor filling one whole dimension of an array is
    written out with the other subscripts kept.

    A full-extent section names the same elements as the bare array does, so
    ``x(:)`` is written exactly as ``x`` is. The second case is the one that
    earns the generality: the section is a rank-1 slice of a rank-3 array, so
    each element assignment has to carry the two subscripts the source fixed.

    '''
    code = '''
        module test
        contains
        subroutine tmp(q)
          integer, intent(in) :: q
          integer :: x(4)
          real :: v(3,2,2)
          x(:) = [1, 2, 3, 4]
          v(:,1,q) = [0.0, 0.0, 1.0]
        end subroutine tmp
        end module test'''
    module = fortran_reader.psyir_from_source(code).children[0].children[0]
    cwriter = CWriter()

    assert cwriter(module[0]) == (
        "x[1] = 1;\n"
        "x[2] = 2;\n"
        "x[3] = 3;\n"
        "x[4] = 4;\n")
    assert cwriter(module[1]) == (
        "v[1 + 1 * vLEN1 + q * vLEN1 * vLEN2] = 0.0;\n"
        "v[2 + 1 * vLEN1 + q * vLEN1 * vLEN2] = 0.0;\n"
        "v[3 + 1 * vLEN1 + q * vLEN1 * vLEN2] = 1.0;\n")


def test_cw_array_constructor_in_an_expression(fortran_reader):
    '''Tests that a constructor used as a value is refused by name.

    Anywhere but the whole right-hand side of an assignment, the constructor
    has to survive as an array in its own right, which needs a temporary this
    backend does not create. The refusal names the position so that a reader
    of the message knows which of the two it is looking at, and says what is
    missing rather than only that the node is unsupported.

    '''
    code = '''
        module test
        contains
        subroutine tmp()
          integer :: x(4)
          real :: y(3)
          x = x + [1, 2, 3, 4]
          x = [ [1, 2], [3, 4] ]
          y = abs([1.0, 2.0, 3.0])
        end subroutine tmp
        end module test'''
    module = fortran_reader.psyir_from_source(code).children[0].children[0]
    cwriter = CWriter()

    for statement, position in ((module[0], "as an operand of an expression"),
                                (module[1], "nested inside another array "
                                            "constructor"),
                                (module[2], "as an argument of the 'ABS' "
                                            "intrinsic")):
        with pytest.raises(VisitorError) as err:
            _ = cwriter(statement)
        assert (f"The C backend cannot write an array constructor {position}: "
                f"C has no array-valued expression, so this constructor needs "
                f"a temporary array to hold its elements and the backend "
                f"creates none." in str(err.value))

    # A constructor whose parent is a call to something other than an
    # intrinsic, and one with no parent at all, are reachable by calling the
    # handler directly, which is how a caller surveying what the writer can
    # write reaches it. Neither is reachable through a whole statement,
    # because this writer has no handler for a Call.
    for parent, position in (
            (Call.create(RoutineSymbol("sub")), "as an actual argument of a "
                                                "call"),
            (None, "in an expression")):
        constructor = ArrayConstructor.create(
            [Literal("1", ScalarType.integer_type())])
        if parent:
            parent.addchild(constructor)
        with pytest.raises(VisitorError) as err:
            _ = cwriter.arrayconstructor_node(constructor)
        assert (f"cannot write an array constructor {position}: "
                in str(err.value))


def test_cw_array_constructor_with_an_implied_do(fortran_reader):
    '''Tests that an implied-do array constructor is refused.

    The refusal is not this back-end\'s. The PSyIR frontend does not model an
    implied do, so ``[ (i, i=1,4) ]`` arrives as a CodeBlock holding the whole
    constructor rather than as an ArrayConstructor with an unusual child, and
    the CodeBlock refusal is the one that fires. This is asserted rather than
    left implicit because the element-by-element translation above would be
    wrong for an implied do -- there is no list of elements to count -- and a
    later change that taught the frontend to model one would silently reach
    that translation.

    '''
    code = '''
        module test
        contains
        subroutine tmp()
          integer :: x(4), i
          x = [ (i, i=1,4) ]
        end subroutine tmp
        end module test'''
    module = fortran_reader.psyir_from_source(code).children[0].children[0]

    assert isinstance(module[0].rhs, CodeBlock)
    with pytest.raises(VisitorError) as err:
        _ = CWriter()(module[0])
    assert "CodeBlocks can not be translated to C." in str(err.value)


def test_cw_array_constructor_needing_a_temporary(fortran_reader):
    '''Tests that a constructor assigned to something other than a whole
    array, or to an array whose origin is not known, is refused.

    A partial section and a rank-2 target are refused because the elements
    would have to be counted against a shape the writer would be guessing at;
    a scalar target because the assignment is not conforming Fortran in the
    first place. An assumed-shape dummy is refused for a different reason:
    the target is right, but its declared lower bound is not in the tree, so
    the Fortran index of an element is not known and an assumed origin of 1
    would be a silently wrong answer wherever it was not 1.

    '''
    code = '''
        module test
        contains
        subroutine tmp(b)
          integer, dimension(:) :: b
          integer :: x(4), s, m(2,2)
          x(2:3) = [1, 2]
          m = [1, 2, 3, 4]
          s = [1]
          b = [1, 2, 3]
        end subroutine tmp
        end module test'''
    module = fortran_reader.psyir_from_source(code).children[0].children[0]
    cwriter = CWriter()

    for statement, name in ((module[0], "x"), (module[1], "m"),
                            (module[2], "s")):
        with pytest.raises(VisitorError) as err:
            _ = cwriter(statement)
        assert (f"The C backend can only write an array constructor into a "
                f"whole rank-1 array or a full-extent section of one "
                f"dimension of an array, but found one assigned to '{name}', "
                f"which is neither: that assignment needs a temporary array "
                f"to hold the constructor." in str(err.value))

    with pytest.raises(VisitorError) as err:
        _ = cwriter(module[3])
    assert ("The C backend cannot write an array constructor into 'b' because "
            "dimension 1 of its declaration has no literal lower bound, so "
            "the Fortran index of each element of the constructor is not "
            "known here." in str(err.value))


# What the writer renders ``x ** n`` as, for the exponents whose answer has
# been measured against gfortran. The strings are the assertion: a tree with
# the same operands in a different association rounds differently, so a test
# that only counted the multiplications would pass on a wrong one.
_INTEGER_POWER_TREES = {
    1: "x",
    2: "(x * x)",
    3: "((x * x) * x)",
    4: "((x * x) * (x * x))",
    5: "(((x * x) * (x * x)) * x)",
    6: "(((x * x) * (x * x)) * (x * x))",
    7: "(((x * x) * (x * x)) * ((x * x) * x))",
    8: "(((x * x) * (x * x)) * ((x * x) * (x * x)))",
    }


def _real_power(exponent):
    '''Build ``x ** exponent`` over a real ``x``.

    :param exponent: the exponent, already a PSyIR node.
    :type exponent: :py:class:`psyclone.psyir.nodes.DataNode`

    :returns: the power operation.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    '''
    return BinaryOperation.create(
        BinaryOperation.Operator.POW,
        Reference(DataSymbol("x", ScalarType.real_type())), exponent)


def test_cw_integer_power_two_is_a_product():
    '''``x ** 2`` is a product rather than a call to 'pow'.

    gfortran does not call the C library for an integer exponent: it
    multiplies, and each multiplication is correctly rounded. 'pow' is not
    required to be, and glibc's is measurably not -- it differs from ``x * x``
    for about one operand in a thousand -- so a region that called it would
    disagree with the Fortran it replaced in the last bit.

    '''
    assert CWriter()(_real_power(
        Literal("2", ScalarType.integer_type()))) == "(x * x)"


def test_cw_integer_power_three_is_a_product_tree():
    '''``x ** 3`` is ``((x * x) * x)``, the tree gfortran builds.

    This is the exponent the C16_MG checksum difference was traced to: 'pow'
    disagrees with the product tree for about a quarter of all operands at
    ``x ** 3``, against one in a thousand at ``x ** 2``, which is why the
    fourth-order kernel drifted where its third-order sibling did not.

    '''
    assert CWriter()(_real_power(
        Literal("3", ScalarType.integer_type()))) == "((x * x) * x)"


@pytest.mark.parametrize("exponent", sorted(_INTEGER_POWER_TREES))
def test_cw_integer_power_up_to_eight(exponent):
    '''Every exponent the writer renders as a tree renders as the measured one.

    Eight is the limit because eight is as far as the comparison against
    gfortran was taken; nine and above stay with 'pow', which is what they
    were.

    '''
    assert CWriter()(_real_power(
        Literal(str(exponent), ScalarType.integer_type()))) == \
        _INTEGER_POWER_TREES[exponent]


def test_cw_integer_power_signed_literal(fortran_reader):
    '''A negative literal exponent is the reciprocal of the tree.

    That is what gfortran does with it, and it is written over the real base
    only: Fortran evaluates an integer raised to a negative power as an
    integer, which is zero for every base but one and minus one, and a
    reciprocal would not be that.

    The numerator is the integer one so that C++'s arithmetic conversions
    give the quotient the base's own type; a '1.0' would compute a
    single-precision reciprocal in double and round it twice.

    An explicitly positive exponent is the same tree as a bare one. The front
    end writes both signs as a unary operation over an unsigned literal, so
    neither reaches the writer as a signed value.

    '''
    code = '''
        module test
        contains
        subroutine tmp()
          real*8 :: a, x
          integer :: i, j
          a = x ** (-2)
          j = i ** (-2)
          a = x ** (+3)
        end subroutine tmp
        end module test'''
    module = fortran_reader.psyir_from_source(code).children[0].children[0]
    cwriter = CWriter()

    assert cwriter(module[0].rhs) == "(1 / (x * x))"
    assert cwriter(module[1].rhs) == "pow(i, (-2))"
    assert cwriter(module[2].rhs) == "((x * x) * x)"


def test_cw_integer_power_real_exponent_stays_pow():
    '''A real exponent is a call to 'pow', which is what Fortran does too.'''
    assert CWriter()(_real_power(
        Literal("3.0", ScalarType.real_type()))) == "pow(x, 3.0)"


def test_cw_integer_power_variable_exponent_stays_pow():
    '''An exponent that is not a literal is a call to 'pow'.

    Its value is not known here, so there is no tree to write; and a zero
    exponent keeps 'pow' as well, whose answer is exactly one.

    '''
    cwriter = CWriter()

    assert cwriter(_real_power(
        Reference(DataSymbol("n", ScalarType.integer_type())))) == "pow(x, n)"
    assert cwriter(_real_power(
        Literal("0", ScalarType.integer_type()))) == "pow(x, 0)"
    assert cwriter(_real_power(
        Literal("9", ScalarType.integer_type()))) == "pow(x, 9)"
