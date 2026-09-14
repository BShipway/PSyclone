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

"""How wide a team the loops a hierarchical region spreads actually need.

The hierarchical launch gives one team to a cell and spreads the region's
chosen loops across that team's members. Until this module existed the team's
width was ``Kokkos::AUTO``, which is 128 members on CUDA and one on OpenMP.
128 is the wrong answer for an LFRic column: the loops spread are over the
levels, a GungHo mesh has 30 of them, so 30 members work, 98 idle, and all 128
rendezvous at every ``team_barrier`` the body carries. Measured on an H100
(phase 7, task B7) the horizontal FFSL flux region fell from 22.6 ms a launch
to 12.1 ms when its team was forced to 32, the warp.

``Kokkos::AUTO`` cannot give that answer, because it knows the policy and not
the loop the region will spread inside it. This module supplies what AUTO
lacks -- the extent of the spread -- as C++ the launch evaluates at run time,
so the team is sized from the mesh the model was configured with rather than
from a table somebody measured on one card.

Only a bound the generated function can evaluate *before* its functor runs is
usable, and that is what :py:func:`spread_extents` selects for. A region's
scalar formals are in scope at that point and nothing else is: the body's own
locals are declared inside the functor, and a bound reading one of them --
``TeamVectorRange(team, b, t + 1)`` is a real GungHo shape -- would not
compile there. Such a loop contributes no extent, and a region none of whose
loops contributes one keeps ``Kokkos::AUTO``; the team is then whatever the
backend chooses, which is what every region got before this module existed.

An excluded loop costs nothing in correctness. A ``TeamVectorRange`` divides
its iterations among however many members the team has, so a team sized from
the loops that could be read runs the loops that could not just as correctly,
in more passes.
"""

from psyclone.psyir.backend.kokkos_region import KokkosScalar
from psyclone.psyir.nodes import (
    BinaryOperation, Literal, Node, Reference, UnaryOperation)


#: The node types an extent expression may be built from, matched by exact
#: class and not by ``isinstance``: an ``ArrayReference`` is a ``Reference``
#: and subscripts a View the functor holds rather than a value in scope where
#: the team is sized. Deliberately narrow -- a bound is admitted because the
#: generated function can evaluate it there, and anything outside arithmetic
#: over literals and scalar formals, a call and a ``CodeBlock`` included,
#: either cannot be evaluated there or would have to be read to find out. A
#: bound this refuses costs a launch nothing but ``Kokkos::AUTO``.
_EXTENT_NODES = (BinaryOperation, Literal, Reference, UnaryOperation)


def _in_launch_scope(expression, scalars):
    """Return whether the generated function can evaluate ``expression``.

    :param expression: one bound of a loop the region spreads.
    :type expression: :py:class:`psyclone.psyir.nodes.Node`
    :param scalars: the names of the region's scalar formals, which are the
        only values in scope where the team is sized.
    :type scalars: Set[str]

    :returns: whether every node of the expression is arithmetic over
        literals and those formals.
    :rtype: bool
    """
    for node in expression.walk(Node):
        if node.__class__ not in _EXTENT_NODES:
            return False
        if isinstance(node, Reference) and node.symbol.name not in scalars:
            return False
    return True


def spread_extents(region, render):
    """Return the extent of each of a region's spread loops, as C++ text.

    One entry per loop whose bounds the generated function can evaluate
    where it sizes the team, in the order the region names them and with
    repeats dropped: the level loop of an LFRic kernel is spread a dozen
    times in one body and its extent is one expression, not a dozen.

    The Fortran bound is inclusive and a Kokkos range is half-open, so the
    extent is ``stop + 1 - start``, which is the count of iterations
    :py:meth:`~psyclone.psyir.backend.kokkos.KokkosWriter.loop_node` hands
    the ``TeamVectorRange``. Both bounds are parenthesised because either
    may be a sum.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param render: the writer's own visitor, which turns one PSyIR
        expression into the C++ the body would have spelt it as. Passed in
        rather than imported so that the extent is rendered by the writer
        generating the region, with its kinds and its Views in force.
    :type render: Callable[
        [:py:class:`psyclone.psyir.nodes.Node`], str]

    :returns: the extent expressions, empty where no loop has usable bounds.
    :rtype: Tuple[str, ...]
    """
    scalars = {argument.name for argument in region.arguments
               if isinstance(argument, KokkosScalar)}
    extents = []
    for loop in region.parallel_loops:
        if not (_in_launch_scope(loop.start_expr, scalars)
                and _in_launch_scope(loop.stop_expr, scalars)):
            continue
        extent = (f"({render(loop.stop_expr)}) + 1 "
                  f"- ({render(loop.start_expr)})")
        if extent not in extents:
            extents.append(extent)
    return tuple(extents)


__all__ = ["spread_extents"]
