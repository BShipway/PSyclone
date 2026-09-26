# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for a team-level write to a whole array, named without a subscript.

Every member of a hierarchical launch executes the statements outside the
loops spread over the team, so an array those statements write is written by
all of them at once, and the writer wraps the write in ``Kokkos::single`` and
follows it with a barrier. ``d = matmul(m, m)`` writes every element of
``d`` and names it with no subscript at all, and until the Phase 7 close the
writer recognised an array write only by its subscript. compute-sanitizer's
racecheck found the consequence in ``compound_operator_kernel``: every member
wrote ``d`` into team scratch, and the spread loop after it read ``d`` with no
barrier between -- a shared-memory race, harmless only because every member
happened to write the same values (Task F2, 2026-09-23).

The region below is that kernel's shape in miniature: a whole-array product
into a team scratch array, and a level loop spread over the team reading it.
"""

from psyclone.psyir.backend.kokkos import (
    KokkosRegion, KokkosScalar, KokkosScratch, KokkosView, KokkosWriter)
from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.nodes import KernelSchedule, Loop, Routine


_OPERATOR_KERNEL = """
subroutine operator_code(nlayers, y, m, ndf, undf, map)
  use constants_mod, only : i_def, r_double
  integer(kind=i_def), intent(in) :: nlayers, ndf, undf
  real(kind=r_double), dimension(undf), intent(inout) :: y
  real(kind=r_double), dimension(ndf, ndf), intent(in) :: m
  integer(kind=i_def), dimension(ndf), intent(in) :: map
  integer(kind=i_def) :: k
  real(kind=r_double) :: scale
  real(kind=r_double), dimension(ndf, ndf) :: d
  scale = 0.5_r_double
  d = matmul(m, m)
  do k = 1, nlayers
    y(map(1) + k - 1) = scale * d(1, 1)
  end do
end subroutine operator_code
"""


def _region():
    """Return the region above, its level loop spread over the team.

    :returns: the region.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    routine = FortranReader().psyir_from_source(
        _OPERATOR_KERNEL).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    schedule = KernelSchedule.create(
        "operator_code", symbol_table=symbol_table, children=children)
    return KokkosRegion(
        name="operator_kokkos",
        schedule=schedule,
        cell_count="ncells",
        arguments=(
            KokkosScalar("nlayers", "int"),
            KokkosView("y", "y_data", "double", ("undf",),
                       index_offsets=(1,)),
            KokkosView("m", "m_data", "double", ("ndf", "ndf"),
                       index_offsets=(1, 1), read_only=True),
            KokkosScalar("ndf", "int"),
            KokkosScalar("undf", "int"),
            KokkosView("map", "map_data", "int", ("ndf", "ncells"),
                       index_offsets=(1,), extra_indices=("cell",),
                       read_only=True, random_access=True),
            KokkosScalar("ncells", "int"),
        ),
        kind_types=(("r_double", "double"), ("i_def", "int")),
        scratch=(KokkosScratch("d", "double", ("ndf", "ndf"),
                               index_offsets=(1, 1)),),
        parallel_loops=(schedule.walk(Loop)[0],))


def test_kokkos_writer_serialises_a_whole_array_write_by_the_team():
    """One member writes the product, and the team waits for it.

    The nest the product is lowered to goes inside one ``Kokkos::single``,
    and a barrier follows it before the spread loop that reads ``d``: without
    the barrier a member could read an element another member is still
    writing, which is the race racecheck reported.
    """
    code = KokkosWriter()(_region())
    single = code.index("Kokkos::single(Kokkos::PerTeam(team), [&]() {\n")
    product = code.index("d((_kae_i0 - 1), (_kae_i1 - 1)) = _kae_r0;")
    barrier = code.index("team.team_barrier();\n", product)
    spread = code.index("Kokkos::TeamVectorRange(team, 1, nlayers + 1)")
    assert single < product < barrier < spread
    # Once in each copy of the launch.
    assert code.count("Kokkos::single(Kokkos::PerTeam(team)") == 2


def test_kokkos_writer_leaves_a_whole_scalar_write_to_every_member():
    """A scalar named without a subscript is still each member's own.

    The whole-array rule reads the symbol's type, not the absence of a
    subscript, so the scalar beside the product is written bare.
    """
    code = KokkosWriter()(_region())
    assert "  scale = 0.5;\n" in code
    assert code.index("scale = 0.5;") < code.index("Kokkos::single(")
