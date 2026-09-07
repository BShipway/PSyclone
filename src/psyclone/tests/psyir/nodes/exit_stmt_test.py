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

''' Performs pytest tests on the Exit PSyIR node. '''

import pytest

from psyclone.errors import GenerationError
from psyclone.psyir.nodes import (
    Assignment, Exit, Literal, Loop, Reference, Routine, WhileLoop)
from psyclone.psyir.nodes.node import colored
from psyclone.psyir.symbols import DataSymbol, ScalarType


def test_exit_node_str():
    ''' Check the node_str method of the Exit class. '''
    exit_stmt = Exit()
    coloredtext = colored("Exit", Exit._colour)
    assert coloredtext+"[]" in exit_stmt.node_str()


def test_exit_can_be_printed():
    '''Test that an Exit instance can always be printed (i.e. is
    initialised fully).'''
    exit_stmt = Exit()
    assert "Exit[]" in str(exit_stmt)


def test_exit_children_validation():
    '''Test that children added to Exit are validated. An Exit node does
    not accept any children.

    '''
    exit_stmt = Exit()
    exit_stmt1 = Exit()
    with pytest.raises(GenerationError) as excinfo:
        exit_stmt.addchild(exit_stmt1)
    assert ("Item 'Exit' can't be child 0 of 'Exit'. Exit is a"
            " LeafNode and doesn't accept children.") in str(excinfo.value)


def test_exit_outside_loop():
    '''Test that an Exit with no ancestor loop is rejected: there is no
    loop for it to terminate, so no back-end can write it.

    '''
    exit_stmt = Exit()
    routine = Routine.create("test", children=[exit_stmt])
    with pytest.raises(GenerationError) as excinfo:
        exit_stmt.validate_global_constraints()
    assert ("Exit must be inside a loop, as it terminates the innermost loop "
            "that encloses it, but this one has no Loop or WhileLoop "
            "ancestor." in str(excinfo.value))
    assert routine is exit_stmt.ancestor(Routine)


def test_exit_inside_loop():
    '''Test that an Exit inside either kind of loop is accepted, however
    deeply it is nested.

    '''
    variable = DataSymbol("i", ScalarType.integer_type())
    exit_stmt = Exit()
    _ = Loop.create(variable, Literal("1", ScalarType.integer_type()),
                    Literal("10", ScalarType.integer_type()),
                    Literal("1", ScalarType.integer_type()), [exit_stmt])
    exit_stmt.validate_global_constraints()

    while_exit = Exit()
    condition = DataSymbol("cond", ScalarType.boolean_type())
    _ = WhileLoop.create(Reference(condition),
                         [Assignment.create(
                             Reference(variable),
                             Literal("1", ScalarType.integer_type())),
                          while_exit])
    while_exit.validate_global_constraints()
