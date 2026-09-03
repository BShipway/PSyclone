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

"""The launch shapes :py:class:`~psyclone.psyir.backend.kokkos.KokkosWriter`
selects among.

Each function here takes a generated body and wraps it in one ``parallel_for``
shape. They are separate from the writer because they are the part of the
back-end that grows as launch shapes are added, while the writer's job -- the
per-node translation of the body and the choice of shape -- does not. The
writer is the only caller; nothing here visits PSyIR.
"""


def range_launch(region, local_declarations, body):
    """Return the ``RangePolicy`` launch, one cell per iteration.

    This is the shape every region had before scratch existed, and it is
    reproduced here unchanged: the captures already in the model are gated
    on whole-model checksums and on assertions over this exact text.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param local_declarations: the generated declarations of the kernel's
        scalar locals, already indented.
    :type local_declarations: str
    :param body: the generated kernel body, already indented.
    :type body: str

    :returns: the ``parallel_for`` and its captured body.
    :rtype: str
    """
    return (
        f'  Kokkos::parallel_for("{region.name}", '
        f"Kokkos::RangePolicy<>(0, {region.cell_count}),\n"
        f"      KOKKOS_LAMBDA(const int {region.cell_index}) {{\n"
        f"{local_declarations}{body}"
        "      });\n")


def team_launch(region, local_declarations, body):
    """Return the ``TeamPolicy`` launch, one cell per team rank.

    Cells are tiled across the ranks of a team so that each rank takes one
    cell and holds its own per-thread scratch. That keeps the parallelism
    identical to :py:func:`range_launch` -- one cell per worker -- while
    giving each worker fast, launch-scoped storage; on a GPU that scratch
    is shared memory rather than global.

    The team size cannot be chosen here, because it depends on how much
    scratch each rank asks for, so the policy is asked for the largest it
    supports. The scratch request is set on the probe policy before the
    query, or the answer is the one for a policy requesting nothing.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param local_declarations: the generated declarations of the kernel's
        scalar locals, already indented.
    :type local_declarations: str
    :param body: the generated kernel body, already indented.
    :type body: str

    :returns: the scratch type aliases, the size computation, the bound
        body, the team-size probe and the ``parallel_for``.
    :rtype: str
    """
    aliases = "".join(
        f"  using {item.name}_scratch_t = Kokkos::View<{item.c_type}"
        f"{'*' * len(item.extents)}, Kokkos::LayoutLeft, ScratchSpace, "
        "Unmanaged>;\n"
        for item in region.scratch)
    sizes = "\n      + ".join(
        f"{item.name}_scratch_t::shmem_size({', '.join(item.extents)})"
        for item in region.scratch)
    constructions = "".join(
        f"      {item.name}_scratch_t {item.name}("
        f"team.thread_scratch(0), {', '.join(item.extents)});\n"
        for item in region.scratch)
    return (
        f"{aliases}\n"
        f"  const size_t scratch_bytes = {sizes};\n\n"
        "  auto body = KOKKOS_LAMBDA(const TeamMember &team) {\n"
        "    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, "
        "team.team_size()),\n"
        "        [&](const int rank) {\n"
        f"      const int {region.cell_index} = team.league_rank() * "
        "team.team_size() + rank;\n"
        # The league is sized by rounding up, so the last team runs with
        # ranks that have no cell. Without this they would run the body
        # for a cell past the end of every View.
        f"      if ({region.cell_index} >= {region.cell_count}) {{\n"
        "        return;\n"
        "      }\n"
        f"{constructions}"
        f"{local_declarations}{body}"
        "    });\n"
        "  };\n\n"
        "  TeamPolicy probe = TeamPolicy(1, Kokkos::AUTO)\n"
        "      .set_scratch_size(0, Kokkos::PerThread(scratch_bytes));\n"
        "  const int team_size = probe.team_size_max(body, "
        "Kokkos::ParallelForTag());\n"
        f"  const int league_size = ({region.cell_count} + team_size - 1)"
        " / team_size;\n"
        f'  Kokkos::parallel_for("{region.name}",\n'
        "      TeamPolicy(league_size, team_size)\n"
        "          .set_scratch_size(0, "
        "Kokkos::PerThread(scratch_bytes)),\n"
        "      body);\n")
