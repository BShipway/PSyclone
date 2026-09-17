# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the kernel-local arrays a team holds one copy each of.

A small kernel-local array no loop spread over the team touches is not team
scratch but per-member storage: every member computes the same values into
its own copy. The back-end's half of that is asserted here -- the wrapper it
declares such an array as, the team scratch it stops reserving for it, and
the ``Kokkos::single`` it stops guarding its writes with. Which arrays
qualify is the transformation's half, and is asserted in
``lfric_kokkos_member_local_test``.

The region below is ``ffsl_flux_xy_panel_remap``'s shape in miniature: a
level loop spread over the team, and beside it a two-element integer array
the team-level code fills and reads. That kernel holds four such arrays, and
holding them in team scratch is what ``nvcc`` 13.3 miscompiled (phase 7,
task W7).
"""

# pylint: disable=protected-access

from dataclasses import replace

import pytest

from psyclone.psyir.backend.kokkos import (
    KokkosRegion, KokkosScalar, KokkosScratch, KokkosView, KokkosWriter)
from psyclone.psyir.backend.kokkos_launch import (
    MEMBER_LOCAL_TYPE, _member_local_declaration, member_local_definition,
    team_scratch_items)
from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.nodes import KernelSchedule, Loop, Routine


# A level loop whose results the team shares in ``column``, and beside it the
# two-element ``local_dofs`` the team-level statements fill and read back.
_MIXED_KERNEL = """
subroutine remap_code(nlayers, y, x, ndf, undf, map)
  use constants_mod, only : i_def, r_double
  integer(kind=i_def), intent(in) :: nlayers, ndf, undf
  real(kind=r_double), dimension(undf), intent(inout) :: y
  real(kind=r_double), dimension(undf), intent(in) :: x
  integer(kind=i_def), dimension(ndf), intent(in) :: map
  integer(kind=i_def) :: k
  integer(kind=i_def), dimension(2) :: local_dofs
  real(kind=r_double), dimension(nlayers) :: column
  local_dofs(1) = 1
  local_dofs(2) = nlayers
  do k = 1, nlayers
    column(k) = x(map(1) + k - 1) * 2.0_r_double
  end do
  y(map(1) + local_dofs(2) - local_dofs(1)) = column(nlayers)
end subroutine remap_code
"""


# The same shape with nothing left for the team to share: the level loop
# writes the field directly, so ``local_dofs`` is the region's only
# kernel-local array and the region reserves no team scratch at all.
_MEMBER_ONLY_KERNEL = """
subroutine index_code(nlayers, y, x, ndf, undf, map)
  use constants_mod, only : i_def, r_double
  integer(kind=i_def), intent(in) :: nlayers, ndf, undf
  real(kind=r_double), dimension(undf), intent(inout) :: y
  real(kind=r_double), dimension(undf), intent(in) :: x
  integer(kind=i_def), dimension(ndf), intent(in) :: map
  integer(kind=i_def) :: k
  integer(kind=i_def), dimension(2) :: local_dofs
  local_dofs(1) = 1
  local_dofs(2) = nlayers
  do k = 1, nlayers
    y(map(1) + k - 1) = x(map(1) + k - 1) * 2.0_r_double
  end do
  y(map(1)) = x(map(1) + local_dofs(2) - local_dofs(1))
end subroutine index_code
"""


def _schedule(source, name):
    """Return the kernel body of one of the sources above.

    :param str source: the Fortran subroutine.
    :param str name: the schedule's name, which is the subroutine's.

    :returns: the body, detached from the file it was parsed from.
    :rtype: :py:class:`psyclone.psyir.nodes.KernelSchedule`
    """
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    return KernelSchedule.create(
        name, symbol_table=symbol_table, children=children)


def _arguments():
    """Return the ABI both regions below are given.

    :returns: the region's arguments, in PSy-layer order.
    :rtype: tuple
    """
    return (
        KokkosScalar("nlayers", "int"),
        KokkosView("y", "y_data", "double", ("undf",), index_offsets=(1,)),
        KokkosView(
            "x", "x_data", "double", ("undf",), index_offsets=(1,),
            read_only=True, random_access=True),
        KokkosScalar("ndf", "int"),
        KokkosScalar("undf", "int"),
        KokkosView(
            "map", "map_data", "int", ("ndf", "ncells"), index_offsets=(1,),
            extra_indices=("cell",), read_only=True, random_access=True),
        KokkosScalar("ncells", "int"),
    )


def _mixed_region(**overrides):
    """Return a region holding one member-local array and one shared one.

    :param overrides: fields to replace on the region.

    :returns: the region.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    schedule = _schedule(_MIXED_KERNEL, "remap_code")
    region = KokkosRegion(
        name="remap_kokkos",
        schedule=schedule,
        cell_count="ncells",
        arguments=_arguments(),
        kind_types=(("r_double", "double"), ("i_def", "int")),
        scratch=(
            KokkosScratch(
                "local_dofs", "int", ("2",), index_offsets=(1,),
                member_local=True),
            KokkosScratch(
                "column", "double", ("nlayers",), index_offsets=(1,)),
        ),
        parallel_loops=(schedule.walk(Loop)[0],))
    return replace(region, **overrides) if overrides else region


def _member_only_region():
    """Return a region whose every kernel-local array is member-local.

    :returns: the region.
    :rtype: :py:class:`psyclone.psyir.backend.kokkos.KokkosRegion`
    """
    schedule = _schedule(_MEMBER_ONLY_KERNEL, "index_code")
    return KokkosRegion(
        name="index_kokkos",
        schedule=schedule,
        cell_count="ncells",
        arguments=_arguments(),
        kind_types=(("r_double", "double"), ("i_def", "int")),
        scratch=(
            KokkosScratch(
                "local_dofs", "int", ("2",), index_offsets=(1,),
                member_local=True),),
        parallel_loops=(schedule.walk(Loop)[0],))


def test_kokkos_scratch_is_shared_unless_said_otherwise():
    """The new field defaults off, so older descriptions are unchanged.

    Every region described before this existed holds team scratch, and the
    generated text of each is asserted elsewhere in the suite; the default
    is what keeps those assertions true.
    """
    item = KokkosScratch("column", "double", ("nlayers",))
    assert item.member_local is False
    assert replace(item, member_local=True).member_local is True


def test_team_scratch_items_leaves_out_the_member_local_arrays():
    """Team scratch is reserved for the shared arrays and no others."""
    assert [item.name for item in team_scratch_items(_mixed_region())] == [
        "column"]
    assert not team_scratch_items(_member_only_region())


def test_member_local_definition_is_emitted_only_where_it_is_used():
    """A region with nothing member-local generates exactly what it did."""
    assert member_local_definition(
        _mixed_region(scratch=(
            KokkosScratch("column", "double", ("nlayers",)),))) == ""
    definition = member_local_definition(_mixed_region())
    assert f"struct {MEMBER_LOCAL_TYPE} {{" in definition
    assert "template <typename T, int N0, int N1 = 1, int N2 = 1>" in \
        definition
    # LayoutLeft's order, leftmost subscript fastest, which is the order the
    # View this replaces had. A wrapper that flattened the other way would
    # compile and give a kernel wrong answers.
    assert "return data[i0 + N0 * i1];" in definition
    assert "return data[i0 + N0 * (i1 + N1 * i2)];" in definition


def test_kokkos_writer_defines_the_wrapper_ahead_of_the_region():
    """The definition is at file scope: a template may not be local."""
    code = KokkosWriter()(_mixed_region())
    assert code.index(f"struct {MEMBER_LOCAL_TYPE} {{") < code.index(
        'extern "C" void remap_kokkos(')


def test_kokkos_writer_declares_a_member_local_array_in_the_functor():
    """It is a declaration where its View construction would have stood."""
    code = KokkosWriter()(_mixed_region())
    assert "    KokkosMemberLocal<int, 2> local_dofs;\n" in code
    # Declared before the shared array, which is the order the region gives
    # its scratch in, which is the kernel's declaration order.
    assert code.index("KokkosMemberLocal<int, 2> local_dofs;") < code.index(
        "column_scratch_t column(")


def test_kokkos_writer_reserves_no_team_scratch_for_a_member_local_array():
    """It contributes no View alias and no term to the size request."""
    code = KokkosWriter()(_mixed_region())
    assert "local_dofs_scratch_t" not in code
    assert "const size_t scratch_bytes = " \
        "column_scratch_t::shmem_size(nlayers);" in code


def test_kokkos_writer_asks_for_no_scratch_where_every_array_is_local():
    """A region left with nothing shared generates as one with no scratch.

    The declarations still open the functor, but the launch reserves
    nothing: there is no size to compute and no request to carry, and the
    scratch memory space is not even named.
    """
    code = KokkosWriter()(_member_only_region())
    assert "    KokkosMemberLocal<int, 2> local_dofs;\n" in code
    assert "scratch_bytes" not in code
    assert "set_scratch_size" not in code
    assert "ScratchSpace" not in code


def test_kokkos_writer_writes_a_member_local_array_without_a_single():
    """Its writes are bare, because there is nothing to serialise.

    Each member writes its own copy, so a ``Kokkos::single`` would leave
    every other member's copy unwritten -- the guard is not merely wasted
    but wrong. The barrier that follows one goes with it, and none is lost:
    a member reads only what it wrote itself.
    """
    code = KokkosWriter()(_mixed_region())
    assert "    local_dofs((1 - 1)) = 1;\n" in code
    assert "    local_dofs((2 - 1)) = nlayers;\n" in code
    # The View write beside them still takes the guard, so what is asserted
    # above is the member-local array and not the launch shape.
    assert "Kokkos::single(Kokkos::PerTeam(team), [&]() {\n      y((" in code
    assert code.count("Kokkos::single(Kokkos::PerTeam(team)") == 1


def test_kokkos_writer_reads_a_member_local_array_like_the_view_it_was():
    """A subscript of it is written as a subscript of any described array.

    This is why the storage is wrapped in a struct rather than declared as
    a plain C array: nothing in the writer that renders a subscript has to
    know which of the two kinds of storage it reached.
    """
    code = KokkosWriter()(_mixed_region())
    assert "local_dofs((2 - 1))) - local_dofs((1 - 1))" in code


@pytest.mark.parametrize("extents,expected", [
    (("2",), "KokkosMemberLocal<int, 2> local_dofs;"),
    (("2", "3"), "KokkosMemberLocal<int, 2, 3> local_dofs;"),
    (("2", "3", "4"), "KokkosMemberLocal<int, 2, 3, 4> local_dofs;"),
])
def test_member_local_declaration_renders_every_rank_it_subscripts(
        extents, expected):
    """One to three extents render as the wrapper's template arguments.

    Asserted on the declaration rather than through a generated region
    because the rank of an array is fixed by the kernel that declares it,
    and one kernel cannot hold the same array at three ranks.
    """
    item = KokkosScratch(
        "local_dofs", "int", extents, index_offsets=(1,) * len(extents),
        member_local=True)
    assert _member_local_declaration(item, "    ") == f"    {expected}\n"
