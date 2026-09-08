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

"""The dof launch shape
:py:class:`~psyclone.psyir.backend.kokkos.KokkosWriter` selects for a region
whose Fortran loop iterated over dofs.

Every shape in :py:mod:`psyclone.psyir.backend.kokkos_launch` launches over
cell columns. An LFRic kernel written for a cell column reads and writes the
dofs of that column through a dofmap, so two columns may reach the same dof
and the launch shape has to say what happens when they do -- which is what
colouring, atomics and the team launches are all about.

A dof kernel is not shaped that way. The Fortran loop it came from runs over
the dofs of a function space directly, and the PSy layer hands the kernel one
dof of each field: ``call kern(f1_data(df), f2_data(df), scalar)``. There is no
dofmap in the call and no cell index in the body. The region carries each of
those per-dof actuals as a rank-1 View sliced to the dof count, exactly as it
carries a per-cell one, and the launch below indexes them with its own
iteration number.

That makes the dof launch the simplest shape the back-end has, and the only one
whose freedom from write conflicts is a property of the iteration space rather
than of something the launch does about it: one iteration writes one dof, so no
two iterations write the same one. It is a separate module from the cell shapes
because it shares the policy type with them and nothing else -- none of the
cell shapes' scratch, team or lower-bound machinery applies to it -- and
because :py:mod:`psyclone.psyir.backend.kokkos` has the least room left of the
three files that could have held it.

**Neither of the two helpers the cell shapes compose is called here, and that
is deliberate rather than an omission.**
:py:func:`~psyclone.psyir.backend.kokkos_launch.launch_index` answers which
name a launch gives its index, and the answer differs from the region's own
only for a coloured region -- which a dof region is not, having no shared write
for a colour map to separate.
The first of those is refused by
:py:meth:`~psyclone.psyir.backend.kokkos.KokkosWriter._validate` rather than
left to be read out of the text below.

:py:func:`~psyclone.psyir.backend.kokkos_launch.launch_offsets` answers where
the counting starts, and a dof loop starts at the first dof: LFRic's ``dof``
and ``owned_dof`` spaces both begin there, so no loop reaching this shape names
a lower bound. A region that named one anyway would launch from zero, the field
being read by none of the text below. That is the shape's stated omission and
not an oversight: it is unused rather than wrong, and the arm that honoured it
would be an untested path for something LFRic does not write.
"""


def dof_launch(region, local_declarations, body):
    """Return the ``RangePolicy`` launch, one dof per iteration.

    The comment the launch carries into the generated source is part of what
    is generated, not a note to a reader of this file: a reviewer of a
    captured region has the C++ and not the PSyIR, and the fact that its
    index is a dof rather than a cell is the one thing about the region that
    the C++ does not otherwise show. Every View the region carries is sliced
    to the dof count and subscripted by that index, which is exactly what a
    per-cell View looks like.

    :param region: the region being generated. Its
        :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_count`
        holds the dof count and its
        :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_index` the
        name of the dof index; the two attributes are named for the cell case
        because a region has one iteration space and one count of it, whether
        that space is cells or dofs.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param local_declarations: the generated declarations of the kernel's
        scalar locals, already indented.
    :type local_declarations: str
    :param body: the generated kernel body, already indented.
    :type body: str

    :returns: the comment stating the iteration space, the ``parallel_for``
        and its captured body.
    :rtype: str
    """
    return (
        "  // One iteration per dof. The Fortran loop this region replaces\n"
        "  // iterated over the dofs of a function space rather than over\n"
        "  // cell columns, so the index below is a position in each of the\n"
        "  // Views the region carries and reaches no dofmap. One iteration\n"
        "  // writes one dof and no two iterations write the same one, so\n"
        "  // this shape needs neither colouring nor an atomic to be safe\n"
        "  // where a launch over cell columns would need one or the other.\n"
        f'  Kokkos::parallel_for("{region.name}", '
        f"Kokkos::RangePolicy<>(0, {region.cell_count}),\n"
        f"      KOKKOS_LAMBDA(const int {region.cell_index}) {{\n"
        f"{local_declarations}{body}"
        "      });\n")


__all__ = ["dof_launch"]
