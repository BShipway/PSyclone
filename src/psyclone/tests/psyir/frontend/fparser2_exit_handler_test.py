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

''' Module containing pytest tests for the handling of the EXIT statement
in the PSyIR fparser2 frontend. '''

import pytest

from fparser.common.readfortran import FortranStringReader
from fparser.two import Fortran2003
from fparser.two.utils import walk

from psyclone.psyir.frontend.fparser2 import Fparser2Reader
from psyclone.psyir.nodes import (
    CodeBlock, Exit, IfBlock, Literal, Loop, Schedule, WhileLoop)
from psyclone.psyir.symbols import DataSymbol, ScalarType
from psyclone.tests.utilities import Compile


def _loop_body():
    '''
    :returns: the body of a counted loop, to stand in for the parent an \
        EXIT statement is handled against.
    :rtype: :py:class:`psyclone.psyir.nodes.Schedule`

    '''
    loop = Loop.create(DataSymbol("i", ScalarType.integer_type()),
                       Literal("1", ScalarType.integer_type()),
                       Literal("10", ScalarType.integer_type()),
                       Literal("1", ScalarType.integer_type()), [])
    return loop.loop_body


def test_exit_in_counted_loop(fortran_reader, fortran_writer, tmpdir):
    ''' An unlabelled EXIT in a counted DO becomes an Exit node. '''
    code = (
        "subroutine test(a)\n"
        "  integer, intent(inout) :: a\n"
        "  integer :: i\n"
        "  do i = 1, 10\n"
        "    if (a > i) then\n"
        "      exit\n"
        "    end if\n"
        "    a = a + i\n"
        "  end do\n"
        "end subroutine test\n")
    psyir = fortran_reader.psyir_from_source(code)
    assert not psyir.walk(CodeBlock)
    exits = psyir.walk(Exit)
    assert len(exits) == 1
    assert exits[0].ancestor(Loop) is psyir.walk(Loop)[0]
    assert exits[0].ancestor(IfBlock) is psyir.walk(IfBlock)[0]

    result = fortran_writer(psyir)
    assert "      exit\n" in result
    assert Compile(tmpdir).string_compiles(result)


def test_exit_in_while_loop(fortran_reader, fortran_writer, tmpdir):
    ''' An unlabelled EXIT in a DO WHILE becomes an Exit node. '''
    code = (
        "subroutine test(a)\n"
        "  integer, intent(inout) :: a\n"
        "  do while (a < 10)\n"
        "    a = a + 1\n"
        "    if (a == 5) exit\n"
        "  end do\n"
        "end subroutine test\n")
    psyir = fortran_reader.psyir_from_source(code)
    assert not psyir.walk(CodeBlock)
    exits = psyir.walk(Exit)
    assert len(exits) == 1
    assert exits[0].ancestor(WhileLoop) is psyir.walk(WhileLoop)[0]

    result = fortran_writer(psyir)
    assert "      exit\n" in result
    assert Compile(tmpdir).string_compiles(result)


def test_exit_in_unconditional_loop(fortran_reader, fortran_writer):
    ''' An EXIT is the only way out of a DO with no loop control, so that
    loop must keep working. '''
    code = (
        "subroutine test(a)\n"
        "  integer, intent(inout) :: a\n"
        "  do\n"
        "    a = a + 1\n"
        "    if (a == 5) exit\n"
        "  end do\n"
        "end subroutine test\n")
    psyir = fortran_reader.psyir_from_source(code)
    assert not psyir.walk(CodeBlock)
    assert len(psyir.walk(Exit)) == 1
    assert "    exit\n" in fortran_writer(psyir)


def test_exit_from_named_construct(fortran_reader, fortran_writer):
    ''' An EXIT that names its construct leaves a loop that is not
    necessarily the innermost one, so the whole DO stays a CodeBlock. '''
    code = (
        "subroutine test(a)\n"
        "  integer, intent(inout) :: a\n"
        "  integer :: i, j\n"
        "  outer: do i = 1, 10\n"
        "    do j = 1, 10\n"
        "      if (a > i) exit outer\n"
        "    end do\n"
        "  end do outer\n"
        "end subroutine test\n")
    psyir = fortran_reader.psyir_from_source(code)
    assert not psyir.walk(Exit)
    assert not psyir.walk(Loop)
    assert len(psyir.walk(CodeBlock)) == 1
    assert "EXIT outer" in fortran_writer(psyir)


def test_exit_handler_refuses_named_construct(parser):
    ''' The handler itself refuses a labelled EXIT by name, whether or not
    the DO handler has already refused the loop that contains it. '''
    code = (
        "program test\n"
        "integer :: i\n"
        "outer: do i = 1, 10\n"
        "  exit outer\n"
        "end do outer\n"
        "end program test\n")
    ast = parser(FortranStringReader(code))
    exit_stmt = walk(ast, Fortran2003.Exit_Stmt)[0]
    processor = Fparser2Reader()
    with pytest.raises(NotImplementedError) as err:
        processor._exit_handler(exit_stmt, _loop_body())
    assert ("EXIT from the named construct 'outer': only an EXIT from the "
            "innermost enclosing loop is supported" in str(err.value))


def test_exit_handler_refuses_exit_with_no_loop(parser):
    ''' An EXIT whose enclosing loop is not in the PSyIR has no loop to
    terminate, so it stays a CodeBlock. '''
    code = (
        "program test\n"
        "integer :: i\n"
        "do i = 1, 10\n"
        "  exit\n"
        "end do\n"
        "end program test\n")
    ast = parser(FortranStringReader(code))
    exit_stmt = walk(ast, Fortran2003.Exit_Stmt)[0]
    processor = Fparser2Reader()
    with pytest.raises(NotImplementedError) as err:
        processor._exit_handler(exit_stmt, Schedule())
    assert ("EXIT with no enclosing loop in the PSyIR" in str(err.value))
