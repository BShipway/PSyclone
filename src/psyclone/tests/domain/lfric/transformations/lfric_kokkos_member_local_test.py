# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosCallMixin: the locals a team need not share.

A kernel-local array is team scratch because the team shares it. Where no
loop the launch spreads over the team ever names the array, and its shape is
a small compile-time constant, it does not have to be: every member computes
the same values into its own copy. This module asserts which arrays the
transformation describes that way; what the back-end then writes is asserted
in ``kokkos_member_local_test``.

The measured reason for drawing the distinction is under phase 7, task W7:
``ffsl_flux_xy_panel_remap`` holds four two-element index arrays no spread
loop touches, and holding them in team scratch is what ``nvcc`` 13.3
miscompiled above ``-Xcicc -O1``.
"""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LEVEL_ALGORITHM, _LEVEL_KERNEL, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.nodes import KernelSchedule, Loop, Routine


# The level kernel holding ``ffsl_flux_xy_panel_remap``'s shape in
# miniature: a two-element index array the team-level statements fill and
# read back, and a level loop that never names it.
_MEMBER_LOCAL_KERNEL = _LEVEL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    integer(kind=i_def), dimension(2) :: local_dofs\n").replace(
    "    scaling = 0.5_r_def\n",
    "    scaling = 0.5_r_def\n"
    "    local_dofs(1) = 1\n"
    "    local_dofs(2) = nlayers\n").replace(
    "    field_out(map_w3(1) + nlayers - 1) = "
    "field_in(map_w3(1) + nlayers - 1)\n",
    "    field_out(map_w3(1) + local_dofs(2) - local_dofs(1)) = &\n"
    "        field_in(map_w3(1) + local_dofs(2) - local_dofs(1))\n")


# The same kernel with the spread loop reading the array. The members of a
# team run that loop between them, so a member reading the array reads a
# copy some other member may have filled: it stays in team scratch.
_TOUCHED_LOCAL_KERNEL = _MEMBER_LOCAL_KERNEL.replace(
    "      field_out(map_w3(1) + k - 1) = scaling * "
    "field_in(map_w3(1) + k - 1)\n",
    "      field_out(map_w3(1) + k - 1) = scaling * &\n"
    "          field_in(map_w3(1) + k - local_dofs(1))\n")


@pytest.fixture(name="member_local_target")
# pylint: disable-next=unused-argument
def member_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel holds a small untouched index array."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _MEMBER_LOCAL_KERNEL)


@pytest.fixture(name="touched_local_target")
# pylint: disable-next=unused-argument
def touched_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose spread loop reads the small index array."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _TOUCHED_LOCAL_KERNEL)


_LOOP_BODY = "    y(map(1) + k - 1) = x(map(1) + k - 1) * 2.0_r_double\n"


def _schedule(declaration, statement="    sized(1) = 1\n",
              loop_body=_LOOP_BODY):
    """Return a level-loop body holding one extra kernel-local array.

    The rule is asked about a schedule directly, rather than through a
    captured invoke, wherever what is asserted is the rule and not its
    plumbing: a shape per threshold through the LFRic front end would parse
    a kernel and an algorithm for each.

    :param str declaration: the Fortran declaration of the array, which the
        tests below vary.
    :param str statement: a team-level statement naming it, so that the
        array is not an unused local.
    :param str loop_body: the body of the loop the launch spreads, which
        names the array in the one test that asks about a spread loop
        touching it.

    :returns: the body, detached from the file it was parsed from.
    :rtype: :py:class:`psyclone.psyir.nodes.KernelSchedule`
    """
    source = f"""
subroutine sized_code(nlayers, y, x, ndf, undf, map)
  use constants_mod, only : i_def, r_double
  integer(kind=i_def), intent(in) :: nlayers, ndf, undf
  real(kind=r_double), dimension(undf), intent(inout) :: y
  real(kind=r_double), dimension(undf), intent(in) :: x
  integer(kind=i_def), dimension(ndf), intent(in) :: map
  integer(kind=i_def) :: k
{declaration}
{statement}
  do k = 1, nlayers
{loop_body}
  end do
end subroutine sized_code
"""
    routine = FortranReader().psyir_from_source(source).walk(Routine)[0]
    symbol_table = routine.symbol_table.detach()
    children = [child.detach() for child in routine.children[:]]
    return KernelSchedule.create(
        "sized_code", symbol_table=symbol_table, children=children)


def _described(declaration, statement="    sized(1) = 1\n",
               loop_body=_LOOP_BODY, spread=True):
    """Return whether the array that declaration declares is member-local.

    :param str declaration: as :py:func:`_schedule` takes it.
    :param str statement: as :py:func:`_schedule` takes it.
    :param str loop_body: as :py:func:`_schedule` takes it.
    :param bool spread: whether the launch spreads the level loop over the
        team, which a region reaching the hierarchical launch does and a
        region reaching either flat shape does not.

    :returns: the ``member_local`` flag of the described array.
    :rtype: bool
    """
    schedule = _schedule(declaration, statement, loop_body)
    loops = tuple(schedule.walk(Loop)) if spread else ()
    described = {item.name: item
                 for item in LFRicKokkosTrans._local_arrays(schedule, loops)}
    return described["sized"].member_local


@pytest.mark.parametrize("declaration", [
    "  integer(kind=i_def), dimension(2) :: sized",
    "  integer(kind=i_def), dimension(2 + 1) :: sized",
    "  real(kind=r_double), dimension(3, 3) :: sized",
    "  integer(kind=i_def), dimension(16) :: sized",
])
def test_a_small_constant_shape_is_described_per_member(declaration):
    """Every shape the rule admits, at the edges of what it admits.

    The literal arithmetic is there because it is how the shapes actually
    occur: ``ffsl_flux_z_rev_nirvana`` declares twelve arrays over
    ``2 + 1``. Sixteen elements is the cap, which has headroom over the
    largest such array in GungHo, a three by three at nine.
    """
    assert _described(declaration) is True


@pytest.mark.parametrize("declaration,reason", [
    ("  integer(kind=i_def), dimension(nlayers) :: sized",
     "an extent the generated code works out at run time"),
    ("  integer(kind=i_def), dimension(17) :: sized",
     "more elements than a copy each is worth"),
    ("  integer(kind=i_def), dimension(2, 2, 2, 2) :: sized",
     "more dimensions than the wrapper subscripts"),
])
def test_a_shape_outside_the_rule_stays_in_team_scratch(declaration, reason):
    """Each threshold refuses on its own, and refusing is the safe answer.

    Team scratch is what every kernel-local array was before this existed,
    so a shape the rule declines is described exactly as it was. ``reason``
    names the threshold under test so that a failure reads as a sentence.
    """
    assert _described(declaration) is False, reason


def test_an_array_a_spread_loop_touches_stays_in_team_scratch():
    """The correctness condition: a member must not read another's copy.

    The members of a team run the spread loop between them, so an array
    read there is read by a member some other member filled the copy for.
    That array is what team scratch is for, and it keeps it.
    """
    declaration = "  integer(kind=i_def), dimension(2) :: sized"
    statement = "    sized(1) = 1\n    sized(2) = nlayers\n"
    assert _described(declaration, statement) is True
    assert _described(
        declaration, statement,
        loop_body="    y(map(1) + k - 1) = "
                  "x(map(1) + k - sized(1))\n") is False


def test_a_region_that_spreads_no_loop_is_described_as_it_was():
    """With no spread loops there is nothing to move and nothing to ask.

    Such a region's members are whole cells rather than lanes of one, so it
    reserves one array per member already. Asserted because the generated
    text of every capture in the model that reaches a flat launch is gated
    elsewhere in this suite, and this is what keeps that text the same.
    """
    assert _described(
        "  integer(kind=i_def), dimension(2) :: sized", spread=False) is False


def test_the_described_region_declares_it_per_member(member_local_target):
    """End to end: the plumbing reaches the generated region.

    The loops the launch spreads are worked out once where the region is
    described and handed to the scratch descriptions, so the array the
    team's members do not share is declared in the functor rather than
    built over team scratch, and no team scratch is reserved for it.
    """
    _, loop, _ = member_local_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "    KokkosMemberLocal<int, 2> local_dofs;\n" in cpp
    assert "local_dofs_scratch_t" not in cpp
    assert "shmem_size" not in cpp
    # Its writes are the team's own work, done by every member on its own
    # copy, so they are bare: a Kokkos::single would leave every other
    # member's copy unwritten.
    assert "    local_dofs((1 - 1)) = 1;\n" in cpp
    assert "    local_dofs((2 - 1)) = nlayers;\n" in cpp


def test_the_described_region_shares_one_a_spread_loop_touches(
        touched_local_target):
    """The same kernel, read inside the spread loop, keeps team scratch."""
    _, loop, _ = touched_local_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "KokkosMemberLocal" not in cpp
    assert "local_dofs_scratch_t local_dofs(team.team_scratch(0), 2);" in cpp
    assert "Kokkos::single(Kokkos::PerTeam(team), [&]() {" in cpp


def test_a_local_the_spread_loop_owns_is_untouched_by_this(local_target):
    """A region with no spread loop generates exactly what it did.

    ``column_solve`` holds two column arrays sized by ``nlayers`` and
    reaches the flat team launch. Neither condition is met, and the
    assertion is that the wrapper appears nowhere in what it generates.
    """
    _, loop, _ = local_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "KokkosMemberLocal" not in cpp
    assert "swept_scratch_t" in cpp


@pytest.mark.parametrize("extent,size", [
    ("2", 2),
    ("(2 + 1)", 3),
    ("(4 + 1) / 2", 2),
    ("(+2)", 2),
    ("(-2)", -2),
    ("2 * 3", 6),
    ("nlayers", None),
    ("nlayers + 1", None),
    ("2.5", None),
    ("MAX(2, 3)", None),
    ("2 / 0", None),
    ("2 +", None),
])
def test_a_size_is_read_from_literal_arithmetic_and_nothing_else(
        extent, size):
    """What an extent has to be for the rule to have a number to test.

    An extent is C++ the generated code evaluates, and this reads the ones
    a compiler would fold: literals, the four operators, and a sign. Where
    the extent names anything -- a kernel argument, a module constant, an
    intrinsic -- there is no number here and the array stays in team
    scratch, which is what sends the six regions whose locals are sized
    from ``nlayers`` or ``ndf`` there. A quotient truncates toward zero, as
    both languages do, and a division by zero answers nothing rather than
    raising: an extent is not this transformation's to evaluate the
    legality of. The last case is not an extent any kernel produces; it is
    there because a parse failure must answer "not a size" rather than
    reach the caller.
    """
    assert LFRicKokkosTrans._member_local_size((extent,)) == size


def test_a_size_is_the_product_of_every_extent():
    """A shape is small enough when its elements are, not its extents."""
    assert LFRicKokkosTrans._member_local_size(("3", "3")) == 9
    assert LFRicKokkosTrans._member_local_size(("4", "nlayers")) is None


def test_an_array_an_alias_is_aimed_at_stays_in_team_scratch():
    """A pointer aimed at it is generated as a View handle.

    There would be no View to hand such a handle, so the array keeps the
    team scratch one, whatever else is true of it. The set of names comes
    from ``_alias_targets``, which ``_local_arrays`` has already asked for
    to leave the pointers themselves out of the scratch it describes.
    """
    schedule = _schedule("  integer(kind=i_def), dimension(2) :: sized")
    symbol = schedule.symbol_table.lookup("sized")
    loops = tuple(schedule.walk(Loop))

    assert LFRicKokkosTrans._is_member_local(
        symbol, ("2",), loops, set()) is True
    assert LFRicKokkosTrans._is_member_local(
        symbol, ("2",), loops, {"sized"}) is False


def test_an_array_of_no_declared_extent_stays_in_team_scratch():
    """An aliasing pointer arrives as an array of none, and is refused.

    ``_local_arrays`` leaves the pointers out before it asks, so this is
    the belt to that braces: an empty shape has no size to test and would
    otherwise multiply out to the empty product, one element.
    """
    schedule = _schedule("  integer(kind=i_def), dimension(2) :: sized")
    symbol = schedule.symbol_table.lookup("sized")

    assert LFRicKokkosTrans._is_member_local(
        symbol, (), tuple(schedule.walk(Loop)), set()) is False
