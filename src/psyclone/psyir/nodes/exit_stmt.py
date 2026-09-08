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

''' This module contains the Exit node implementation.'''

from psyclone.errors import GenerationError
from psyclone.psyir.nodes.loop import Loop
from psyclone.psyir.nodes.statement import Statement
from psyclone.psyir.nodes.while_loop import WhileLoop


class Exit(Statement):
    '''
    Node representing the termination of the innermost enclosing loop:
    Fortran's unlabelled ``EXIT``, C's ``break``.

    A loop is the whole of this node's meaning -- it names no loop, and
    which loop it leaves is decided by where it sits -- so it is only valid
    with a :py:class:`psyclone.psyir.nodes.Loop` or
    :py:class:`psyclone.psyir.nodes.WhileLoop` ancestor. An ``EXIT`` that
    leaves a named construct further out is not this node; the frontend
    leaves that one in a
    :py:class:`psyclone.psyir.nodes.CodeBlock`.

    '''
    # Textual description of the node.
    _children_valid_format = "<LeafNode>"
    _text_name = "Exit"
    _colour = "yellow"

    def validate_global_constraints(self):
        '''
        Check that this Exit has a loop to terminate.

        A transformation is free to move statements between schedules, and
        one that moved an Exit out of its loop would otherwise reach a
        back-end that has nothing to write: Fortran's EXIT and C's break are
        both errors outside a loop.

        :raises GenerationError: if this node has no ancestor loop.

        '''
        if self.ancestor((Loop, WhileLoop)) is None:
            raise GenerationError(
                "Exit must be inside a loop, as it terminates the innermost "
                "loop that encloses it, but this one has no Loop or WhileLoop "
                "ancestor.")
        super().validate_global_constraints()
