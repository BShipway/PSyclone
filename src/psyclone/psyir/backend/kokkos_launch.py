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


def launch_offsets(region):
    """Return the three pieces of text a region's lower bound contributes.

    An LFRic loop over the halo cells alone begins where the owned cells end,
    so its launch cannot begin at zero. The first cell it does begin at is a
    scalar formal the region takes, named by
    :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_start`, and
    each of the three shapes below needs it in a different place: the range
    shape as the policy's own lower bound, and the two team shapes as an
    offset on the cell they compute from a league rank together with a league
    shortened by the cells that are being skipped.

    A region with no lower bound gets ``"0"``, no offset and the count
    unchanged, which is the text every shape wrote before this field existed.
    That is asserted rather than reasoned about: the captures already in the
    model are gated on whole-model checksums and on assertions over this
    exact text.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the launch's first index; the offset a shape adds to an index
        it derives from a league rank, empty where there is none; and the
        number of cells the launch covers.
    :rtype: Tuple[str, str, str]
    """
    if region.cell_start is None:
        return "0", "", region.cell_count
    return (region.cell_start, f"{region.cell_start} + ",
            f"({region.cell_count} - {region.cell_start})")


def _scratch_text(region, allocation, indent):
    """Return the three pieces of C++ a region's scratch arrays generate.

    The two team launches place scratch differently -- one array per rank in
    the flat shape, one per team in the hierarchical one -- but the aliases
    and the size sum are the same text in both, and were duplicated between
    them until this function held them. ``allocation`` and ``indent`` are the
    whole of the difference.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param allocation: the member function the Views are constructed over,
        ``team.thread_scratch(0)`` or ``team.team_scratch(0)``.
    :type allocation: str
    :param indent: the leading whitespace of each construction, which differs
        because the flat shape nests its body one level deeper.
    :type indent: str

    :returns: the type aliases, the ``shmem_size`` sum, and the View
        constructions.
    :rtype: Tuple[str, str, str]
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
        f"{indent}{item.name}_scratch_t {item.name}("
        f"{allocation}, {', '.join(item.extents)});\n"
        for item in region.scratch)
    return aliases, sizes, constructions


def scratch_guard(region):
    """Return the C++ that stops a divided scratch extent going negative.

    A kernel-local array may be declared over an integer division --
    ``real(kind=r_tran), dimension((stencil_size + 1) / 2) :: u_local_x_1``
    is a GungHo shape -- and
    :py:func:`~psyclone.psyir.backend.kokkos.is_extent` admits one because
    Fortran and C++ round an integer quotient the same way: both truncate
    toward zero. The generated size is therefore the size the kernel
    declared.

    Two things are still owed to a reader of the generated source. The first
    is that sentence, which is emitted as a comment rather than left to be
    known: a scratch size is one of the few places where a rounding rule
    silently taken on trust would produce a wrong allocation rather than a
    compile error. The second is the case neither language defines, an
    allocation of negative size, which is stopped where it is requested
    instead of being handed to ``shmem_size``.

    Nothing is emitted for a region whose scratch does not divide, so every
    region generated before division was admitted generates what it did then.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the comment and one guard per dividing extent, or the empty
        string where no scratch extent divides.
    :rtype: str
    """
    divided = [(item.name, extent) for item in region.scratch
               for extent in item.extents if "/" in extent]
    if not divided:
        return ""
    checks = "".join(
        f"  if (({extent}) < 0) {{\n"
        f'    Kokkos::abort("{region.name}: scratch extent for '
        f"'{name}' is negative\");\n"
        "  }\n"
        for name, extent in divided)
    return (
        "  // A scratch extent below divides. Fortran and C++ both\n"
        "  // truncate an integer quotient toward zero, so the sizes\n"
        "  // computed here are the sizes the kernel declared. What\n"
        "  // neither language defines is an allocation of negative size,\n"
        "  // so that is stopped rather than requested.\n"
        f"{checks}")


def range_launch(region, local_declarations, body):
    """Return the ``RangePolicy`` launch, one cell per iteration.

    This is the shape every region had before scratch existed, and a region
    with no first cell is reproduced here unchanged: the captures already in
    the model are gated on whole-model checksums and on assertions over this
    exact text.

    A region that names a first cell begins the policy there instead of at
    zero, which is what a loop over the halo alone needs; see
    :py:func:`launch_offsets`.

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
    first, _, _ = launch_offsets(region)
    return (
        f'  Kokkos::parallel_for("{region.name}", '
        f"Kokkos::RangePolicy<>({first}, {region.cell_count}),\n"
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

    Where the region names a first cell the league covers the cells from
    there to the count and each rank adds it back, so the shape's
    parallelism is unchanged and only the cells it visits move; see
    :py:func:`launch_offsets`.

    The probe policy carries the scratch request and is asked for the team
    size the backend recommends for a ``parallel_for`` of this functor. On
    the OpenMP backend that recommendation is one thread, so the leagues
    rather than the ranks carry the parallelism and no team runs more than
    one cell in sequence. The request is set on the probe before the query,
    or the answer is the one for a policy asking for nothing -- a backend
    other than OpenMP may read it.

    ``team_size_max`` is not asked instead, although the largest team the
    scratch permits sounds like the accommodating answer. On OpenMP it
    returns the whole thread pool whatever the scratch request, which put
    one team on the whole league with a ``team_rendezvous`` between
    consecutive cells; the largest team the scratch permits is not the team
    that runs fastest.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param local_declarations: the generated declarations of the kernel's
        scalar locals, already indented.
    :type local_declarations: str
    :param body: the generated kernel body, already indented.
    :type body: str

    :returns: the scratch type aliases, :py:func:`scratch_guard`, the size
        computation, the bound body, the team-size probe and the
        ``parallel_for``.
    :rtype: str
    """
    aliases, sizes, constructions = _scratch_text(
        region, "team.thread_scratch(0)", "      ")
    _, offset, span = launch_offsets(region)
    return (
        f"{aliases}\n"
        f"{scratch_guard(region)}"
        f"  const size_t scratch_bytes = {sizes};\n\n"
        "  auto body = KOKKOS_LAMBDA(const TeamMember &team) {\n"
        "    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, "
        "team.team_size()),\n"
        "        [&](const int rank) {\n"
        f"      const int {region.cell_index} = {offset}team.league_rank()"
        " * team.team_size() + rank;\n"
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
        "  const int team_size = probe.team_size_recommended(body, "
        "Kokkos::ParallelForTag());\n"
        f"  const int league_size = ({span} + team_size - 1)"
        " / team_size;\n"
        f'  Kokkos::parallel_for("{region.name}",\n'
        "      TeamPolicy(league_size, team_size)\n"
        "          .set_scratch_size(0, "
        "Kokkos::PerThread(scratch_bytes)),\n"
        "      body);\n")


def hierarchical_launch(region, local_declarations, body):
    """Return the ``TeamPolicy`` launch, one team per cell.

    The league carries the cells and the team carries the levels: each team
    takes one cell, and the loops the region named in
    :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.parallel_loops` are
    spread across its members by
    :py:meth:`~psyclone.psyir.backend.kokkos.KokkosWriter.loop_node`. That is
    the opposite division from :py:func:`team_launch`, and it is the one an
    LFRic kernel is shaped for: a cell's levels are the inner dimension of
    every field it reads, so members of one team touch neighbouring elements
    rather than columns a stride apart.

    Where the region names a first cell the league is shortened to the cells
    from there to the count and each team adds it back to its rank, so one
    team still takes one cell; see :py:func:`launch_offsets`.

    The team's own work -- the scalars, the loop control, and the boundary
    writes ``Kokkos::single`` guards -- is what a cell has that its levels do
    not, so it sits in the functor rather than in a range over the league.
    Every member runs it redundantly on its own copy of the locals.

    :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.team_size` renders
    as a literal in the policy rather than as anything the generated code
    computes, so a run may be given a different team by regenerating nothing
    but this line -- which is what the forced-team build does to reach the
    team-level concurrency ``Kokkos::AUTO`` sizes to one member on a host.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param local_declarations: the generated declarations of the kernel's
        scalar locals, already indented.
    :type local_declarations: str
    :param body: the generated kernel body, already indented.
    :type body: str

    :returns: the scratch type aliases, :py:func:`scratch_guard` and the
        size computation where the region has scratch, and the
        ``parallel_for`` over one team per cell.
    :rtype: str
    """
    aliases, sizes, constructions = _scratch_text(
        region, "team.team_scratch(0)", "    ")
    preamble = (
        f"{aliases}\n{scratch_guard(region)}"
        f"  const size_t scratch_bytes = {sizes};\n\n"
        if region.scratch else "")
    team_size = (
        "Kokkos::AUTO" if region.team_size is None
        else str(region.team_size))
    _, offset, span = launch_offsets(region)
    policy = f"TeamPolicy({span}, {team_size})" + (
        "\n          .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes))"
        if region.scratch else "")
    return (
        f"{preamble}"
        f'  Kokkos::parallel_for("{region.name}",\n'
        f"      {policy},\n"
        "      KOKKOS_LAMBDA(const TeamMember &team) {\n"
        f"    const int {region.cell_index} = {offset}team.league_rank();\n"
        f"{constructions}{local_declarations}{body}"
        "  });\n")
