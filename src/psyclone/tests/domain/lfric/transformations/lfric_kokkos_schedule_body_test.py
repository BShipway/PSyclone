# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for preparing a selected schedule's body for capture."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _HDIV_SECTION_KERNEL, _LEVEL_ALGORITHM, _LEVEL_KERNEL, _SECTION_ALGORITHM,
    _SECTION_KERNEL, _invoke)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import Exit


# The tri_solve shape as the model actually writes it: a forward elimination
# and a backward substitution, each carrying its own recurrence, and no third
# loop. _LOCAL_KERNEL above is this kernel with the backward sweep's result
# copied out in a separate loop, which the dependence analysis accepts -- so
# _LOCAL_KERNEL takes the hierarchical launch and this one does not. That is
# what keeps a witness for the flat team launch: sci_tri_solve_kernel_mod has
# no parallelisable loop at all, and a fixture that says so has to be shaped
# like it rather than like a kernel with one.
_TRI_SOLVE_ALGORITHM = """
program kokkos_tri_solve_test
  use field_mod, only : field_type
  use tri_sweep_kernel_mod, only : tri_sweep_kernel_type
  implicit none
  type(field_type) :: y_vec, x_vec
  call invoke(tri_sweep_kernel_type(y_vec, x_vec))
end program kokkos_tri_solve_test
"""


_TRI_SOLVE_KERNEL = """
module tri_sweep_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: tri_sweep_kernel_type
    type(arg_type) :: meta_args(2) = (/                              &
         arg_type(gh_field, gh_real, gh_write, w3),                  &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: tri_sweep_code
  end type tri_sweep_kernel_type
contains
  subroutine tri_sweep_code(nlayers, y, x, ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    real(kind=r_def), dimension(undf_w3), intent(inout) :: y
    real(kind=r_def), dimension(undf_w3), intent(in) :: x
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, ij
    real(kind=r_def), dimension(nlayers) :: x_new, tri_plus_new
    real(kind=r_def) :: denom
    k = 0
    ij = map_w3(1)
    denom = 1.0_r_def / x(ij + k)
    tri_plus_new(1) = x(ij + k) * denom
    x_new(1) = x(ij + k) * denom
    do k = 1, nlayers - 1
      denom = 1.0_r_def / (x(ij + k) - x(ij + k) * tri_plus_new(k))
      tri_plus_new(k + 1) = x(ij + k) * denom
      x_new(k + 1) = (x(ij + k) - x(ij + k) * x_new(k)) * denom
    end do
    k = nlayers - 1
    y(ij + k) = x_new(k + 1)
    do k = nlayers - 2, 0, -1
      y(ij + k) = x_new(k + 1) - tri_plus_new(k + 1) * y(ij + k + 1)
    end do
  end subroutine tri_sweep_code
end module tri_sweep_kernel_mod
"""


# The same kernel with a second parallelisable loop inside the first. One team
# is one pool of members, so nesting a TeamVectorRange inside another would
# divide the same members twice; the outer loop alone is taken.
_NESTED_LEVEL_KERNEL = _LEVEL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k, df\n").replace(
    "    real(kind=r_def) :: scaling",
    "    real(kind=r_def) :: scaling\n"
    "    real(kind=r_def), dimension(nlayers,4) :: work").replace(
    "      field_out(map_w3(1) + k - 1) = "
    "scaling * field_in(map_w3(1) + k - 1)",
    "      do df = 1, 4\n"
    "        work(k, df) = scaling * field_in(map_w3(1) + k - 1)\n"
    "      end do\n"
    "      field_out(map_w3(1) + k - 1) = work(k, 1)")


# The same kernel with the level loop stepped. The dependence analysis accepts
# it -- the iterations are as independent as they were -- and the launch still
# refuses it, because TeamVectorRange(team, begin, end) counts by one and has
# no stride to give it.
_STEPPED_LEVEL_KERNEL = _LEVEL_KERNEL.replace(
    "    do k = 1, nlayers - 1", "    do k = 1, nlayers - 1, 2")


# The same kernel with an EXIT leaving the level loop. C++ has no way to leave
# a lambda's enclosing loop, so a spread loop cannot hold the break the EXIT
# becomes; the loop is left serial and the region keeps the flat launch.
_EXIT_LEVEL_KERNEL = _LEVEL_KERNEL.replace(
    "      field_out(map_w3(1) + k - 1) = "
    "scaling * field_in(map_w3(1) + k - 1)",
    "      if (field_in(map_w3(1) + k - 1) < 0.0_r_def) exit\n"
    "      field_out(map_w3(1) + k - 1) = "
    "scaling * field_in(map_w3(1) + k - 1)")


# The nested kernel with the EXIT in the inner loop instead. That break stays
# inside a serial `for` in the lambda body, which is legal C++, so the outer
# loop may still be spread.
_INNER_EXIT_LEVEL_KERNEL = _NESTED_LEVEL_KERNEL.replace(
    "        work(k, df) = scaling * field_in(map_w3(1) + k - 1)",
    "        if (field_in(map_w3(1) + k - 1) < 0.0_r_def) exit\n"
    "        work(k, df) = scaling * field_in(map_w3(1) + k - 1)")


# A local set from an array constructor, the way poly1d_reconstruction writes
# its face list. The values are positional, so the statement is a section that
# must not be lowered: the C writer spreads a constructor over its destination
# itself, and a loop would leave a subscripted constructor behind.
_CONSTRUCTOR_SECTION_KERNEL = _SECTION_KERNEL.replace(
    "    integer(kind=i_def) :: nl, w3_idx, b_idx\n",
    "    integer(kind=i_def) :: nl, w3_idx, b_idx\n"
    "    integer(kind=i_def), dimension(4) :: faces\n").replace(
    "    difference(w3_idx : w3_idx + nl) = &\n"
    "        mass_flux(b_idx + 1 : b_idx + nl + 1) "
    "- mass_flux(b_idx : b_idx + nl)",
    "    faces(:) = (/ 2_i_def, 3_i_def, 4_i_def, 5_i_def /)\n"
    "    difference(w3_idx : w3_idx + nl) = "
    "mass_flux(b_idx : b_idx + nl) * real(faces(1), r_tran)")


@pytest.fixture(name="tri_solve_target")
# pylint: disable-next=unused-argument
def tri_solve_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel is two recurrences and nothing else."""
    return _invoke(
        tmp_path, "tri_sweep", _TRI_SOLVE_ALGORITHM, _TRI_SOLVE_KERNEL)


@pytest.fixture(name="nested_level_target")
# pylint: disable-next=unused-argument
def nested_level_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel nests one level loop inside another."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _NESTED_LEVEL_KERNEL)


@pytest.fixture(name="exit_level_target")
# pylint: disable-next=unused-argument
def exit_level_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel leaves its level loop with an EXIT."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _EXIT_LEVEL_KERNEL)


@pytest.fixture(name="inner_exit_level_target")
# pylint: disable-next=unused-argument
def inner_exit_level_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel leaves an inner loop with an EXIT."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _INNER_EXIT_LEVEL_KERNEL)


@pytest.fixture(name="stepped_level_target")
# pylint: disable-next=unused-argument
def stepped_level_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose level loop counts in twos."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _STEPPED_LEVEL_KERNEL)


def test_lfric_kokkos_trans_captures_an_array_section(section_target):
    """A whole-column section is lowered to a loop and then generated.

    The finite-volume family writes a column as one assignment, which the C
    writer has no Range handler for. The section is therefore lowered before
    the region is described, using PSyclone's own
    ArrayAssignment2LoopsTrans, rather than teaching the Kokkos backend a
    second way to say the same thing.
    """
    psy, loop, _ = section_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert 'extern "C" void fv_difference_kokkos(' in cpp
    # The section became a counted loop over the column, keeping each side's
    # own lower bound as an offset from the assigned range's start. The
    # lowering is a producer of loops, and the loop it produces is one the
    # dependence analysis accepts, so the column is spread over the team
    # rather than swept by one member of it.
    assert ("Kokkos::parallel_for(Kokkos::TeamVectorRange"
            "(team, w3_idx, (w3_idx + nl) + 1)," in cpp)
    assert "difference((idx - 1)) = " in cpp
    assert "mass_flux(((idx + ((b_idx + 1) - w3_idx)) - 1))" in cpp
    assert "mass_flux(((idx + (b_idx - w3_idx)) - 1))" in cpp
    # The counter the lowering introduced is declared inside the lambda. It
    # names no Fortran kind, so it reaches the C writer's own default rather
    # than a kind the region described -- which is the fallback in
    # KokkosWriter.gen_declaration, load-bearing rather than defensive.
    assert "int idx;" in cpp
    assert "TeamPolicy(ncells, Kokkos::AUTO)" in cpp
    assert "Kokkos::RangePolicy" not in cpp

    assert "call fv_difference_kokkos(" in fortran
    assert "call fv_difference_code(" not in fortran


def test_lfric_kokkos_trans_accepts_a_section_actual(
        tmp_path, clear_module_manager_instance):
    """The convert_hdiv_native family's array statements are all lowered.

    Three shapes stand between that family and a capture, and this kernel is
    the three of them without the routine they surround. ``vector = 0.0`` is
    a whole array named with no accessor at all, which
    ArrayAssignment2LoopsTrans refuses for want of one and
    Reference2ArrayRangeTrans supplies; ``vector(:,i) = vector(:,i) + ...``
    is a written section of a local; and the field write takes a column of
    that local by section. What is left of the family after this is its call,
    which is a capability of its own.
    """
    # pylint: disable=unused-argument
    psy, loop, _ = _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM, _HDIV_SECTION_KERNEL)

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert 'extern "C" void fv_difference_kokkos(' in cpp
    # The local is a rank-2 automatic, so it is team scratch rather than a
    # temporary of the lowering's own: the nests write through the View the
    # region already described.
    assert "vector_scratch_t vector(team.team_scratch(0), nlayers, 3);" in cpp
    # 'vector = 0.0' became a nest over both of its dimensions, the second
    # spread over the team and the first -- the contiguous one under
    # LayoutLeft -- swept inside it.
    assert "vector((idx_1 - 1), (idx - 1)) = 0.0;" in cpp
    # The written section keeps the kernel's own loop as the parallel one and
    # counts the column inside it, each side from its own lower bound.
    assert ("vector((idx_2 - 1), (i - 1)) = (vector((idx_2 - 1), (i - 1)) + "
            "mass_flux(((idx_2 + (b_idx - 1)) - 1)));" in cpp)
    # The field write is a column of the local read at the assigned range's
    # offset, which is what says the two sections were matched up rather than
    # subscripted independently.
    assert ("difference((idx_3 - 1)) = "
            "vector(((idx_3 + (1 - w3_idx)) - 1), (1 - 1));" in cpp)
    assert "call fv_difference_kokkos(" in fortran
    assert "call fv_difference_code(" not in fortran


def test_lfric_kokkos_trans_keeps_an_array_constructor_whole(
        tmp_path, clear_module_manager_instance):
    """A constructor assigned to a section is written, not lowered.

    ``faces(:) = (/ 2, 3, 4, 5 /)`` is a section by its left-hand side and a
    list of values by its right, and the two do not survive being separated:
    a loop over the section would have to subscript the constructor, which
    nothing can render. CWriter already spreads a constructor over its
    destination element by element, so the lowering leaves this statement
    alone and that path is reached.
    """
    # pylint: disable=unused-argument
    _, loop, _ = _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM,
        _CONSTRUCTOR_SECTION_KERNEL)

    cpp = LFRicKokkosTrans().apply(loop)

    assert "faces((1 - 1)) = 2;" in cpp
    assert "faces((4 - 1)) = 5;" in cpp
    # Written once for the whole team rather than by every member, and by no
    # loop: a counted loop over the four values is exactly what must not
    # appear.
    assert "Kokkos::single(Kokkos::PerTeam(team)" in cpp
    assert "faces((idx" not in cpp
    # The section that is not a constructor is lowered as it always was, so
    # the two live in one kernel without either changing the other.
    assert "difference((idx - 1)) = (mass_flux(" in cpp


def test_lfric_kokkos_trans_keeps_the_flat_launch_for_scalar_locals(
        solver_target):
    """A kernel whose only locals are scalars keeps the RangePolicy shape.

    ``scaled_solver_code`` declares ``k``, ``df`` and ``scaled`` and no array,
    so admitting local arrays must not have made every region a team launch.
    """
    _, loop, kernel = solver_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._local_arrays(schedule) == ()

    cpp = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::RangePolicy<>(0, ncells)" in cpp
    assert "TeamPolicy" not in cpp
    assert "thread_scratch" not in cpp
    # The scalar local is still declared in the body, where it always was.
    assert "float scaled;" in cpp


def test_parallel_loops_selects_a_level_loop_the_analysis_accepts(
        level_target):
    """A level loop whose iterations are independent is selected.

    The judgement is PSyclone's own: ``_parallel_loops`` asks
    ``DependencyTools.can_loop_be_parallelised`` rather than deciding for
    itself what a level loop is. This one writes ``field_out`` at ``map_w3(1)
    + k - 1`` and reads ``field_in`` at the same place, so no iteration reads
    what another wrote.
    """
    _, _, kernel = level_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    selected = LFRicKokkosTrans._parallel_loops(schedule)

    assert [loop.variable.name for loop in selected] == ["k"]


def test_parallel_loops_refuses_a_recurrence(tri_solve_target):
    """A tridiagonal sweep has no loop that may be spread.

    Both of ``tri_sweep_code``'s loops read at ``k - 1`` or ``k + 1`` what a
    neighbouring iteration wrote, which is the shape the whole selection
    exists to keep out of a ``TeamVectorRange``. The kernel is a miniature of
    ``sci_tri_solve_kernel_mod``, one of the six captured regions, so this is
    the case the model actually contains rather than an invented one.
    """
    _, _, kernel = tri_solve_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._parallel_loops(schedule) == ()


def test_parallel_loops_takes_the_outermost_of_a_nested_pair(
        nested_level_target):
    """Only the outer loop of an acceptable nest is selected.

    ``TeamVectorRange`` may not be nested inside itself, and spreading the
    outer loop already occupies the team, so an inner loop that would also
    qualify is left as a serial loop inside the lambda.
    """
    _, _, kernel = nested_level_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    selected = LFRicKokkosTrans._parallel_loops(schedule)

    assert [loop.variable.name for loop in selected] == ["k"]


def test_parallel_loops_skips_a_stepped_loop(stepped_level_target):
    """A loop with a step other than one is left alone.

    ``TeamVectorRange(team, start, stop + 1)`` gives every member a unit
    stride, so a stepped loop would have to be rewritten before it could be
    spread. It is skipped instead, and the analysis is not even asked: the
    step is checked first, so a stepped loop the analysis would accept is
    still skipped.
    """
    _, _, kernel = stepped_level_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._parallel_loops(schedule) == ()


def test_parallel_loops_skips_a_loop_an_exit_leaves(exit_level_target):
    """A loop an EXIT leaves is left alone however independent it is.

    The body of a spread loop is a lambda, and C++ has no break that leaves
    the loop a lambda was launched over: `break` there is either a compile
    error or leaves something else. The analysis would accept this loop --
    it is the level loop of `test_parallel_loops_selects_a_level_loop...`
    with one statement added -- so the refusal has to be the shape rule it
    is rather than a dependence.
    """
    _, _, kernel = exit_level_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert schedule.walk(Exit)
    assert LFRicKokkosTrans._parallel_loops(schedule) == ()


def test_parallel_loops_keeps_a_loop_an_inner_exit_leaves(
        inner_exit_level_target):
    """An EXIT from an inner loop does not stop the outer one being spread.

    The break it becomes sits inside a serial `for` in the lambda body,
    which compiles and means what the Fortran meant. Only the loop the EXIT
    actually leaves is disqualified.
    """
    _, _, kernel = inner_exit_level_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    selected = LFRicKokkosTrans._parallel_loops(schedule)

    assert [loop.variable.name for loop in selected] == ["k"]


def test_apply_writes_break_for_an_exit(exit_level_target):
    """The captured region leaves its loop with a break.

    The EXIT is a node the C writer knows, so nothing about the capture is
    special-cased for it; what the region has to show is the break in the
    serial loop the selection left serial.
    """
    psy, loop, _ = exit_level_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "break;" in cpp
    assert "TeamVectorRange" not in cpp
    assert "call column_scale_kokkos(" in str(psy.gen)


def test_parallel_loops_classifies_the_lowered_section(section_target):
    """The loop the section lowering produces is classified like any other.

    ``fv_difference_code`` has no loop at all in its source: its column is
    one array assignment, and ``_lower_sections`` turns that into a loop. The
    selection therefore has to run after the lowering, which is why
    ``validate`` predicts on a lowered copy rather than on the schedule as
    parsed.
    """
    _, _, kernel = section_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._parallel_loops(schedule) == ()

    LFRicKokkosTrans._lower_sections(schedule)
    LFRicKokkosTrans._substitute_bounds(schedule)
    selected = LFRicKokkosTrans._parallel_loops(schedule)

    assert [loop.variable.name for loop in selected] == ["idx"]


def test_apply_builds_a_hierarchical_region_for_a_level_loop(level_target):
    """A kernel with a parallel level loop takes the hierarchical launch.

    No scratch is involved: the selection that reaches the hierarchical shape
    is the loop one, and it reaches it on its own. The league is one team per
    cell, so the cell index comes from the league rank rather than from a
    flat rank folded onto cells.
    """
    psy, loop, _ = level_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "TeamPolicy(ncells, Kokkos::AUTO)," in cpp
    assert "KOKKOS_LAMBDA(const TeamMember &team) {" in cpp
    assert "const int cell = team.league_rank();" in cpp
    assert ("Kokkos::parallel_for(Kokkos::TeamVectorRange"
            "(team, 1, (nlayers - 1) + 1)," in cpp)
    assert "        [&](const int k) {" in cpp
    assert "team.team_barrier();" in cpp
    # The trailing scalar write is outside the spread loop, so one member
    # makes it and the rest wait.
    assert "Kokkos::single(Kokkos::PerTeam(team), [&]() {" in cpp
    # Nothing asked for scratch, so nothing sizes any.
    assert "set_scratch_size" not in cpp
    assert "ScratchSpace" not in cpp
    assert "Kokkos::RangePolicy" not in cpp

    assert "call column_scale_kokkos(" in str(psy.gen)


def test_apply_keeps_tri_solve_flat(tri_solve_target):
    """A kernel with scratch and no parallel loop keeps the flat launch.

    This is the witness that the two selections are separate. Scratch alone
    reaches the flat team launch, where the team is a way of owning a
    per-column allocation rather than a way of sharing the column's work, so
    the scratch is per member and the league is a flat range of ranks folded
    onto cells.
    """
    _, loop, _ = tri_solve_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "team_size_recommended(body, Kokkos::ParallelForTag())" in cpp
    assert "x_new_scratch_t x_new(team.thread_scratch(0), nlayers);" in cpp
    assert "if (cell >= ncells) {\n        return;\n      }" in cpp
    assert "TeamVectorRange" not in cpp
    assert "Kokkos::single" not in cpp
    # The flat launch does read the league rank -- it folds it and the member
    # rank into one flat index -- so its absence is not what distinguishes
    # the two. The cell is derived rather than being the league rank itself.
    assert "const int cell = team.league_rank();" not in cpp
    assert "team.league_rank() * team.team_size() + rank;" in cpp
