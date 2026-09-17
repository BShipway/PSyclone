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


def launch_index(region):
    """Return the name a launch gives the index it iterates over.

    Each shape below counts from :py:func:`launch_offsets`' first index to
    :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_count`, and
    for all but one region that count is the mesh cells and the index is
    :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.cell_index`. A
    region captured from a coloured loop counts the cells of one colour
    instead, and the mesh cell is looked up from that through its
    :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.colour_map`; the
    writer emits that lookup as the region's first local declaration, so
    what each shape here has to change is only the name it declares.

    This function answers *what the index is called*; :py:func:`launch_offsets`
    answers *where the counting starts and how far it runs*. The two are
    independent of each other at every site, and the one region that could
    not compose them -- a colour map together with a first cell, whose index
    counts one colour's cells while the bound counts the mesh's -- is refused
    by
    :py:meth:`~psyclone.psyir.backend.kokkos.KokkosWriter._validate` rather
    than generated.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the launch's own index name.
    :rtype: str
    """
    if region.colour_map is None:
        return region.cell_index
    return region.colour_map.index


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

    The name the offset is applied to is :py:func:`launch_index`'s, which is
    the composition of the two: a halo launch of an uncoloured region offsets
    the region's own cell index, and no region reaching here both names a
    colour map and begins past the first cell.

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


def global_scratch_names(region):
    """Return the names of the scratch arrays an alias handle may be aimed at.

    These are placed in level-1 team scratch -- global memory -- rather than
    in level 0, which the CUDA backend puts in shared memory. The reason is
    a code-generation defect met on 2026-09-14 (phase 7, task B6b): an
    ``AnonymousSpace`` handle aimed at a shared-memory scratch array in one
    branch and at a global argument View in the other is a pointer the
    compiler has to keep generic, and ``nvcc`` 13.3 does not. It infers
    "shared" for the merged pointer, converts the global pointer with
    ``cvta.to.shared`` and reads through ``ld.shared``, so the branch that
    aims the handle at the argument reads a garbage shared-memory offset
    (``ffsl_flux_z_nirvana``, ``field_ptr`` under ``log_space``; a
    sixty-line Kokkos kernel reproduces it). With every target of a handle
    in global memory the merged pointer is generic on both sides and the
    defect has nothing to specialise. The cost is one column of global
    memory per team for those arrays alone; every other scratch array stays
    in shared memory.

    Only scratch arrays are returned: an alias target that is an argument
    View is already in global memory and needs no moving.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the names of the scratch arrays some alias names as a target.
    :rtype: Set[str]
    """
    targets = {name for alias in region.aliases for name in alias.targets}
    return {item.name for item in region.scratch if item.name in targets}


def _scratch_level(region, item):
    """Return the scratch level a scratch array is placed in, as text.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param item: the scratch array.
    :type item: :py:class:`psyclone.psyir.backend.kokkos.KokkosScratch`

    :returns: ``"1"`` for an alias target (see
        :py:func:`global_scratch_names`), ``"0"`` otherwise.
    :rtype: str
    """
    return "1" if item.name in global_scratch_names(region) else "0"


#: The name of the generated wrapper a member-local array is declared as.
MEMBER_LOCAL_TYPE = "KokkosMemberLocal"


def team_scratch_items(region):
    """Return the scratch arrays the team shares, member-local ones aside.

    These are the arrays the launch reserves team scratch for. A region
    whose every scratch array is member-local reserves none and asks for
    none, which is why callers ask this rather than testing
    :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.scratch`.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the shared scratch arrays, in the order the region gives them.
    :rtype: Tuple[
        :py:class:`psyclone.psyir.backend.kokkos.KokkosScratch`, ...]
    """
    return tuple(item for item in region.scratch if not item.member_local)


def member_local_definition(region):
    """Return the wrapper definition a member-local array is declared as.

    A kernel-local array the team does not have to share is held by every
    member instead: it is declared inside the functor, so each member's copy
    is its own, and no team scratch is reserved for it. Which arrays those
    are is decided where the region is described, by
    ``LFRicKokkosCallMixin._is_member_local``.

    The storage is wrapped in a struct rather than declared as a plain C
    array so that a subscript of it is written exactly as a subscript of the
    View it replaces: ``x(i)`` and ``x(i, j)`` are what
    :py:meth:`~psyclone.psyir.backend.kokkos_array_expression_mixin.\
KokkosArrayExpressionMixin.arrayreference_node` emits for any described
    array, and nothing else in the writer has to know which of the two kinds
    of storage it reached. The elements are ordered as ``LayoutLeft`` orders
    them, leftmost subscript fastest, which is the order the View had.

    Nothing is emitted for a region with no member-local array, so every
    region generated before this existed is generated as it was then.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the definition and the blank line after it, or the empty
        string.
    :rtype: str
    """
    if not any(item.member_local for item in region.scratch):
        return ""
    return (
        "// A kernel-local array small enough for every member of a team to\n"
        "// hold its own copy, and which no loop spread over the team ever\n"
        "// touches, is held here rather than shared in team scratch: the\n"
        "// members all compute the same values into it, so sharing one buys\n"
        "// nothing and costs a Kokkos::single and a barrier at every write.\n"
        "// Elements are ordered as LayoutLeft ordered the View's.\n"
        "template <typename T, int N0, int N1 = 1, int N2 = 1>\n"
        f"struct {MEMBER_LOCAL_TYPE} {{\n"
        "  T data[N0 * N1 * N2];\n"
        "  KOKKOS_INLINE_FUNCTION T &operator()(int i0) { return data[i0]; }\n"
        "  KOKKOS_INLINE_FUNCTION T &operator()(int i0, int i1) {\n"
        "    return data[i0 + N0 * i1];\n"
        "  }\n"
        "  KOKKOS_INLINE_FUNCTION T &operator()(int i0, int i1, int i2) {\n"
        "    return data[i0 + N0 * (i1 + N1 * i2)];\n"
        "  }\n"
        "};\n\n")


def _member_local_declaration(item, indent):
    """Return the declaration of one member-local array.

    :param item: the scratch array, which ``member_local`` is set on.
    :type item: :py:class:`psyclone.psyir.backend.kokkos.KokkosScratch`
    :param str indent: the leading whitespace, as :py:func:`_scratch_text`
        uses it.

    :returns: the declaration line.
    :rtype: str
    """
    return (f"{indent}{MEMBER_LOCAL_TYPE}<{item.c_type}, "
            f"{', '.join(item.extents)}> {item.name};\n")


def _scratch_text(region, allocation, indent):
    """Return the four pieces of C++ a region's scratch arrays generate.

    The two team launches place scratch differently -- one array per rank in
    the flat shape, one per team in the hierarchical one -- but the aliases
    and the size sum are the same text in both, and were duplicated between
    them until this function held them. ``allocation`` and ``indent`` are the
    whole of the difference.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param allocation: the member function the Views are constructed over,
        ``team.thread_scratch(0)`` or ``team.team_scratch(0)``; the level in
        it is rewritten to ``1`` for the arrays
        :py:func:`global_scratch_names` places in global memory.
    :type allocation: str
    :param indent: the leading whitespace of each construction, which differs
        because the flat shape nests its body one level deeper.
    :type indent: str

    A member-local array is not in team scratch at all, so it contributes
    no alias and no size: it appears among the constructions alone, as the
    declaration :py:func:`member_local_definition` describes, in the place
    its View construction would have stood.

    :returns: the type aliases, the level-0 ``shmem_size`` sum (``0`` where
        every array moved to level 1 or to a member), the level-1 sum (empty
        where none did), and the constructions.
    :rtype: Tuple[str, str, str, str]
    """
    shared = team_scratch_items(region)
    aliases = "".join(
        f"  using {item.name}_scratch_t = Kokkos::View<{item.c_type}"
        f"{'*' * len(item.extents)}, Kokkos::LayoutLeft, ScratchSpace, "
        "Unmanaged>;\n"
        for item in shared)
    by_level = {"0": [], "1": []}
    for item in shared:
        by_level[_scratch_level(region, item)].append(
            f"{item.name}_scratch_t::shmem_size({', '.join(item.extents)})")
    sizes = "\n      + ".join(by_level["0"]) or "0"
    sizes_global = "\n      + ".join(by_level["1"])
    constructions = "".join(
        _member_local_declaration(item, indent) if item.member_local else
        f"{indent}{item.name}_scratch_t {item.name}("
        f"{allocation.replace('(0)', f'({_scratch_level(region, item)})')}, "
        f"{', '.join(item.extents)});\n"
        for item in region.scratch)
    return aliases, sizes, sizes_global, constructions


def global_scratch_size(sizes_global):
    """Return the level-1 size computation, or nothing where none is needed.

    :param str sizes_global: the level-1 ``shmem_size`` sum from
        :py:func:`_scratch_text`, empty where no array moved.

    :returns: the comment and the ``scratch_bytes_1`` computation.
    :rtype: str
    """
    if not sizes_global:
        return ""
    return (
        "  // The arrays below are targets of an alias handle and live in\n"
        "  // level-1 (global) team scratch: a handle aimed at shared memory\n"
        "  // in one branch and at a global View in the other is a pointer\n"
        "  // nvcc 13.3 wrongly specialises to shared memory.\n"
        f"  const size_t scratch_bytes_1 = {sizes_global};\n")


def global_scratch_policy(sizes_global, per):
    """Return the level-1 request to append to a ``TeamPolicy``.

    :param str sizes_global: the level-1 sum, empty where no array moved.
    :param str per: ``PerThread`` or ``PerTeam``, matching the launch.

    :returns: the ``.set_scratch_size(1, ...)`` text, or nothing.
    :rtype: str
    """
    if not sizes_global:
        return ""
    return f"\n          .set_scratch_size(1, Kokkos::{per}(scratch_bytes_1))"


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
    A member-local array is not asked either: its shape is a compile-time
    constant, so a run-time check that it is not negative is dead code.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the comment and one guard per dividing extent, or the empty
        string where no scratch extent divides.
    :rtype: str
    """
    divided = [(item.name, extent) for item in team_scratch_items(region)
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
        f"      KOKKOS_LAMBDA(const int {launch_index(region)}) {{\n"
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
    aliases, sizes, sizes_global, constructions = _scratch_text(
        region, "team.thread_scratch(0)", "      ")
    _, offset, span = launch_offsets(region)
    level_one = global_scratch_policy(sizes_global, "PerThread")
    return (
        f"{aliases}\n"
        f"{scratch_guard(region)}"
        f"  const size_t scratch_bytes = {sizes};\n"
        f"{global_scratch_size(sizes_global)}\n"
        "  auto body = KOKKOS_LAMBDA(const TeamMember &team) {\n"
        "    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, "
        "team.team_size()),\n"
        "        [&](const int rank) {\n"
        f"      const int {launch_index(region)} = {offset}team.league_rank()"
        " * team.team_size() + rank;\n"
        # The league is sized by rounding up, so the last team runs with
        # ranks that have no cell. Without this they would run the body
        # for a cell past the end of every View.
        f"      if ({launch_index(region)} >= {region.cell_count}) {{\n"
        "        return;\n"
        "      }\n"
        f"{constructions}"
        f"{local_declarations}{body}"
        "    });\n"
        "  };\n\n"
        "  TeamPolicy probe = TeamPolicy(1, Kokkos::AUTO)\n"
        "      .set_scratch_size(0, Kokkos::PerThread(scratch_bytes))"
        f"{level_one};\n"
        "  const int team_size = probe.team_size_recommended(body, "
        "Kokkos::ParallelForTag());\n"
        f"  const int league_size = ({span} + team_size - 1)"
        " / team_size;\n"
        f'  Kokkos::parallel_for("{region.name}",\n'
        "      TeamPolicy(league_size, team_size)\n"
        "          .set_scratch_size(0, "
        f"Kokkos::PerThread(scratch_bytes)){level_one},\n"
        "      body);\n")


def _team_scratch(region):
    """Return the three pieces of scratch text the hierarchical launch needs.

    The launch's own policy and the probe that clamps its team both carry
    the scratch request, and the request has to be the same text in both:
    a probe asking for no scratch answers for a policy that is not the one
    being launched.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`

    :returns: the type aliases, guard and size computation that precede the
        launch; the ``.set_scratch_size`` text a policy carries; and the
        View constructions that open the functor. The first two are empty
        for a region with no scratch, which is what such a region generated
        before scratch existed.
    :rtype: Tuple[str, str, str]
    """
    aliases, sizes, sizes_global, constructions = _scratch_text(
        region, "team.team_scratch(0)", "    ")
    if not team_scratch_items(region):
        return "", "", constructions
    return (
        f"{aliases}\n{scratch_guard(region)}"
        f"  const size_t scratch_bytes = {sizes};\n"
        f"{global_scratch_size(sizes_global)}\n",
        "\n          .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes))"
        + global_scratch_policy(sizes_global, "PerTeam"),
        constructions)


def _largest_extent(extents):
    """Return the C++ for the largest of a region's spread extents.

    :param extents: the extent expressions, from
        :py:func:`~psyclone.psyir.backend.kokkos_spread_extent.spread_extents`
        and never empty.
    :type extents: Tuple[str, ...]

    :returns: the single extent, or a nest of ``std::max`` over all of them.
    :rtype: str
    """
    largest = extents[-1]
    for extent in reversed(extents[:-1]):
        largest = f"std::max({extent}, {largest})"
    return largest


def computed_team_size(extents, scratch_request):
    """Return the run-time team size the hierarchical launch is given.

    ``Kokkos::AUTO`` is 128 members on CUDA whatever the region does with
    them, and an LFRic region's members are the levels of one column: a
    GungHo mesh has 30, so 98 of the 128 idle through the body and all 128
    rendezvous at each ``team_barrier`` it carries. Measured on an H100
    (phase 7, task B7) the horizontal FFSL flux region fell from 22.6 ms a
    launch to 12.1 ms when its team was cut to 32. This computes that
    answer instead of taking it from a table: the spread extent, rounded up
    to a whole warp, clamped to the largest team the backend will run this
    functor with, and clamped below at one member.

    Rounded **up** to a warp because a GPU schedules a warp whether or not
    its lanes have work, so a team of 30 idles the same two lanes a team of
    32 does and buys nothing; and clamped to ``team_size_max`` rather than
    to ``team_size_recommended`` because the recommendation answers a
    different question -- the occupancy the backend would like -- while what
    is wanted here is only that a team the spread asked for is one the
    backend will actually launch.

    **Only a GPU backend computes anything.** On OpenMP ``Kokkos::AUTO`` is
    one member per team, the leagues carry the parallelism, and a
    ``TeamPolicy`` asking for more members than the thread pool holds is
    refused at launch -- so a host build, which the whole-model checksum
    gates run on one thread, keeps ``AUTO`` and the behaviour it had before
    this was written. The guard is on the backend rather than on anything
    the region says, because that is where the difference is.

    :param extents: the extent expressions of the loops this region spreads,
        never empty; see
        :py:func:`~psyclone.psyir.backend.kokkos_spread_extent.spread_extents`.
    :type extents: Tuple[str, ...]
    :param scratch_request: the ``.set_scratch_size`` text the launch's own
        policy carries, empty for a region with no scratch. The probe
        carries it too, or it answers for a policy asking for no scratch and
        the clamp is taken against a team the real launch could not run.
    :type scratch_request: str

    :returns: the guarded declaration of ``team_size``.
    :rtype: str
    """
    return (
        "  // The team is sized from the loops this region spreads rather\n"
        "  // than left to Kokkos::AUTO, which knows the policy and not the\n"
        "  // loop: AUTO is 128 members on CUDA, while the spread below is\n"
        "  // typically one column of levels. Rounded up to a warp, since a\n"
        "  // partial warp idles its remaining lanes anyway, and clamped to\n"
        "  // the widest team this functor can be launched with.\n"
        "#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)\n"
        "#if defined(KOKKOS_ENABLE_HIP)\n"
        "  constexpr int warp_width = 64;\n"
        "#else\n"
        "  constexpr int warp_width = 32;\n"
        "#endif\n"
        f"  const int spread_extent = {_largest_extent(extents)};\n"
        f"  TeamPolicy probe = TeamPolicy(1, Kokkos::AUTO){scratch_request};\n"
        "  const int team_max = probe.team_size_max(body, "
        "Kokkos::ParallelForTag());\n"
        "  const int team_size = std::max(1,\n"
        "      std::min(((spread_extent + warp_width - 1) / warp_width)\n"
        "                   * warp_width, team_max));\n"
        "#else\n"
        "  // On a host backend AUTO is one member a team, which is what the\n"
        "  // leagues of this shape are sized for; a team wider than the\n"
        "  // thread pool is refused at launch.\n"
        "  const auto team_size = Kokkos::AUTO;\n"
        "#endif\n")


def hierarchical_launch(region, local_declarations, body, extents=()):
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

    The team's width is one of three things, in this order.
    :py:attr:`~psyclone.psyir.backend.kokkos.KokkosRegion.team_size` renders
    as a literal in the policy rather than as anything the generated code
    computes, so a run may be given a different team by regenerating nothing
    but this line -- which is what the forced-team build does to reach the
    team-level concurrency ``Kokkos::AUTO`` sizes to one member on a host,
    and what an application profile does to force a measured answer.
    Failing that, a region whose spread extent the generated function can
    evaluate sizes its own team from it on a GPU backend; see
    :py:func:`computed_team_size`. Failing both, the team is
    ``Kokkos::AUTO``, as every region's was before the extent was computed.

    Only the computed shape names its functor. The other two keep the
    lambda where it always was, inline in the ``parallel_for``, so that a
    region reaching either of them generates exactly the text it did before
    this: the captures already in the model are gated on assertions over
    that text. The computed shape has to name it, because the clamp asks
    the backend how wide a team it will run *this functor* with, and a
    functor cannot be asked about before it exists.

    :param region: the region being generated.
    :type region: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    :param local_declarations: the generated declarations of the kernel's
        scalar locals, already indented.
    :type local_declarations: str
    :param body: the generated kernel body, already indented.
    :type body: str
    :param extents: the extents of the loops the region spreads, as C++ the
        generated function can evaluate; empty where none of them can be,
        which is the fall back to ``Kokkos::AUTO``. See
        :py:func:`~psyclone.psyir.backend.kokkos_spread_extent.spread_extents`.
    :type extents: Tuple[str, ...]

    :returns: the scratch type aliases, :py:func:`scratch_guard` and the
        size computation where the region has scratch, the team size where
        it is computed, and the ``parallel_for`` over one team per cell.
    :rtype: str
    """
    preamble, scratch_request, constructions = _team_scratch(region)
    computed = region.team_size is None and bool(extents)
    _, offset, span = launch_offsets(region)
    width = "team_size" if computed else (
        "Kokkos::AUTO" if region.team_size is None else region.team_size)
    policy = f"TeamPolicy({span}, {width}){scratch_request}"
    functor = (
        f"    const int {launch_index(region)} = "
        f"{offset}team.league_rank();\n"
        f"{constructions}{local_declarations}{body}")
    launch = (f'  Kokkos::parallel_for("{region.name}",\n'
              f"      {policy},\n")
    if not computed:
        return (
            f"{preamble}{launch}"
            "      KOKKOS_LAMBDA(const TeamMember &team) {\n"
            f"{functor}"
            "  });\n")
    return (
        f"{preamble}"
        "  auto body = KOKKOS_LAMBDA(const TeamMember &team) {\n"
        f"{functor}"
        "  };\n\n"
        f"{computed_team_size(extents, scratch_request)}"
        f"{launch}"
        "      body);\n")
