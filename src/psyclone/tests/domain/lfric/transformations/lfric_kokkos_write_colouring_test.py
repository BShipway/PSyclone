# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the colouring answer to a shared write."""

# pylint: disable=protected-access

import re

import pytest

from lfric_kokkos_sources import (
    _HALO_CELL_ALGORITHM, _HALO_CELL_KERNEL, _LEVEL_ALGORITHM, _SECOND_KERNEL,
    _SHARED_WRITE_ALGORITHM, _SHARED_WRITE_KERNEL, _coloured_inner, _formals,
    _invoke)

from psyclone.domain.lfric import LFRicKern, LFRicLoop
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.errors import GenerationError
from psyclone.psyir.transformations import TransformationError
from psyclone.transformations import LFRicColourTrans


# A sibling the capture cannot take and colouring must: 'gh_inc' onto W2, a
# continuous space, which is what makes 'colour_loops' pick it up. Every
# captured invoke before stage 10 left colouring nothing to do, so the
# ordering fault this shape exposes stayed latent.
_COLOURED_KERNEL = """
module assemble_w2_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_inc, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w2, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: assemble_w2_kernel_type
    type(arg_type) :: meta_args(2) = (/                       &
         arg_type(gh_field, gh_real, gh_inc,  w2),            &
         arg_type(gh_field, gh_real, gh_read, w3) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: assemble_w2_code
  end type assemble_w2_kernel_type
contains
  subroutine assemble_w2_code(nlayers, field_out, field_in, &
                              ndf_w2, undf_w2, map_w2, &
                              ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w2, undf_w2
    integer(kind=i_def), intent(in) :: ndf_w3, undf_w3
    real(kind=r_def), dimension(undf_w2), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    integer(kind=i_def), dimension(ndf_w2), intent(in) :: map_w2
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w2
        field_out(map_w2(df) + k) = field_in(map_w3(1) + k)
      end do
    end do
  end subroutine assemble_w2_code
end module assemble_w2_kernel_mod
"""


# The shape 'apply_split_mixed_operator' has in GungHo: one loop the
# transformation captures, and one beside it that stays Fortran and is
# coloured afterwards.
_COLOURED_ALGORITHM = """
program kokkos_coloured_test
  use constants_mod, only : r_tran
  use field_mod, only : field_type
  use scaled_copy_kernel_mod, only : scaled_copy_kernel_type
  use assemble_w2_kernel_mod, only : assemble_w2_kernel_type
  implicit none
  type(field_type) :: out_field, in_field, vel_field
  real(kind=r_tran) :: scaling
  call invoke(scaled_copy_kernel_type(out_field, in_field, scaling),  &
              assemble_w2_kernel_type(vel_field, out_field))
end program kokkos_coloured_test
"""


@pytest.fixture(name="coloured_target")
# pylint: disable-next=unused-argument
def coloured_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose captured loop has a continuous-space sibling."""
    return _invoke(
        tmp_path, "scaled_copy", _COLOURED_ALGORITHM, _SECOND_KERNEL,
        extra={"assemble_w2_kernel_mod": _COLOURED_KERNEL})


def test_lfric_kokkos_trans_leaves_later_colouring_initialised(
        coloured_target):
    """Colouring applied after a capture must still reach the PSy layer.

    The capture materialises the PSy-layer symbols early, because
    'KernCallArgList' needs them specialised. A transformation running
    afterwards can add loops -- colouring replaces one loop with two -- and
    those loops carry placeholder bounds until the invoke initialises them.
    Suppressing that initialisation leaves the placeholders in the Fortran,
    where they are undeclared, and leaves the colour map declared but never
    assigned. The second is the dangerous one: a dangling 'cmap' compiles.

    """
    psy, loop, _ = coloured_target

    LFRicKokkosTrans().apply(loop)
    schedule = psy.invokes.invoke_list[0].schedule
    sibling = [each for each in schedule.walk(LFRicLoop) if each.kernels()][0]
    LFRicColourTrans().apply(sibling)
    fortran = str(psy.gen)

    assert "uninitialised_loop" not in fortran
    assert "ncolour = mesh%get_ncolours()" in fortran
    assert "cmap => mesh%get_colour_map()" in fortran
    assert ("last_halo_cell_all_colours = "
            "mesh%get_last_halo_cell_all_colours()" in fortran)
    # Everything the coloured loop reads is assigned before it runs.
    assert (fortran.index("cmap => mesh%get_colour_map()") <
            fortran.index("do colour ="))


def test_lfric_kokkos_trans_later_colouring_keeps_captured_bounds(
        coloured_target):
    """The completion pass binds the new loops and leaves the old ones.

    Re-binding a loop the first pass already handled would leave its first
    pair of bounds assigned and unread, and would repoint the region call at
    a second copy of a cell count it already has.

    """
    psy, loop, _ = coloured_target

    LFRicKokkosTrans().apply(loop)
    schedule = psy.invokes.invoke_list[0].schedule
    sibling = [each for each in schedule.walk(LFRicLoop) if each.kernels()][0]
    LFRicColourTrans().apply(sibling)
    fortran = str(psy.gen)

    assert "loop0_stop = mesh%get_last_edge_cell()" in fortran
    assert fortran.count("loop0_stop = ") == 1
    assert fortran.count("! Set-up all of the loop bounds") == 1
    assert "map_wtheta, loop0_stop)" in fortran


def test_lfric_kokkos_trans_refuses_colouring_of_a_meshless_invoke(
        tmp_path, clear_module_manager_instance):
    # pylint: disable=unused-argument
    """An invoke with no mesh object cannot have its colouring completed.

    'LFRicColourTrans' creates the mesh symbol when the invoke had none, but
    only the PSy-layer set-up assigns it, and that reads a kernel argument
    which the capture has by then removed from the tree. Refusing says so;
    emitting the look-up anyway would dereference a null pointer at runtime.

    """
    psy, loop, _ = _invoke(
        tmp_path, "scaled_copy", _COLOURED_ALGORITHM, _SECOND_KERNEL,
        extra={"assemble_w2_kernel_mod": _COLOURED_KERNEL}, dist_mem=False)

    LFRicKokkosTrans().apply(loop)
    schedule = psy.invokes.invoke_list[0].schedule
    sibling = [each for each in schedule.walk(LFRicLoop) if each.kernels()][0]
    LFRicColourTrans().apply(sibling)

    with pytest.raises(GenerationError) as err:
        _ = psy.gen
    assert "has been coloured after its PSy-layer symbols were set up" in str(
        err.value)
    assert "'mesh' is never assigned" in str(err.value)


def test_lfric_kokkos_trans_colours_a_continuous_write_reading_its_target(
        continuous_read_back_target):
    """The refusal's advice is advice that works.

    Colouring runs the cells that meet at a dof in different launches, so
    the read and the store are one cell's own and the statement needs no
    shape at all -- which is why the coloured arm asks nothing of it.
    """
    psy, loop, _ = continuous_read_back_target
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)

    cpp = LFRicKokkosTrans().apply(_coloured_inner(schedule))

    assert "atomic" not in cpp
    assert re.search(r"^\s*flux\(.*\) = ", cpp, re.MULTILINE)


def test_lfric_kokkos_trans_accepts_a_coloured_loop(shared_write_target):
    """A loop colouring has already made safe is captured without atomics.

    This is the second of the two answers to a shared write, and the one
    LFRic's own OpenMP path takes: the cells of one colour share no dof, so
    the launch over them races with nothing and the update is a plain
    read-modify-write. The colours are run one after another, which is where
    the ordering that makes it safe comes from, so the capture takes the
    inner loop and leaves the Fortran loop over colours to make one launch
    per colour.

    The colour map crosses the ABI because the region's cell is no longer its
    launch index: the launch counts the cells of this colour and the map says
    which cell of the mesh each of those is.

    """
    psy, loop, _ = shared_write_target
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)

    cpp = LFRicKokkosTrans().apply(_coloured_inner(schedule))
    fortran = str(psy.gen)

    # One launch per colour: the call sits inside the surviving Fortran loop
    # over colours, and the loop over cells is gone.
    assert "do colour = loop0_start, loop0_stop, 1" in fortran
    assert "call inc_probe_kokkos(" in fortran
    assert fortran.index("do colour =") < fortran.index(
        "call inc_probe_kokkos(")
    assert "call inc_probe_code(" not in fortran

    # The colour map and the colour itself are formals of the region, and
    # the PSy layer passes both.
    call_line = [line for line in fortran.splitlines()
                 if "call inc_probe_kokkos(" in line][0]
    assert "cmap" in call_line
    assert "colour" in call_line
    assert "const int *cmap_data" in cpp
    assert "const int colour" in cpp
    assert "cmap_data" in _formals(cpp)
    assert "colour" in _formals(cpp)
    # The region's cell is the map's answer rather than its launch index.
    assert "cmap(colour - 1" in cpp

    # And no atomic anywhere: that is the whole point of the arm.
    assert "atomic" not in cpp


def test_lfric_kokkos_trans_colours_a_loop_that_takes_an_operator(
        shared_write_operator_target):
    """A coloured operator kernel reads its cell through the colour map.

    LFRic supplies the cell index as an actual argument for exactly the
    kernels that take an operator, because the kernel does arithmetic with it
    -- ``ik = (cell - 1) * nlayers + 1`` -- to find its own slice of the local
    stencil. Uncoloured, that actual is the loop's own variable. Coloured, it
    is ``cmap(colour, cell)``, and it must be recognised as the same thing:
    the region declares the cell position from its own cell rather than taking
    it, so a refusal here would refuse 'matrix_vector', the commonest shared
    write GungHo has.

    The two declarations are ordered, and that is the whole of the fix: the
    colour map answers first, and the one-based position is derived from its
    answer rather than from the launch index, which counts the cells of one
    colour and is not a cell of the mesh.

    """
    psy, loop, _ = shared_write_operator_target
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)

    cpp = LFRicKokkosTrans().apply(_coloured_inner(schedule))

    # 'cell' is the kernel's own formal, so the launch index generated clear
    # of it is 'cell_1', and the position the kernel is given is the map's
    # answer plus one rather than the launch index plus one.
    assert "const int cell_1 = cmap(colour - 1, cell_in_colour) - 1;" in cpp
    assert "const int cell = cell_1 + 1;" in cpp
    assert cpp.index("const int cell_1 =") < cpp.index("const int cell =")
    assert "cell" not in _formals(cpp)
    assert "atomic" not in cpp


def test_lfric_kokkos_trans_refuses_colouring_and_atomics_together(
        shared_write_target):
    """Asking for both answers to one shared write is refused, not ranked.

    Either arm is correct on its own and the two together are correct as
    well, at the cost of an atomic on data that colouring has already made
    private to one launch. A transformation that quietly dropped one of them
    would decide by preference something the caller had stated, so the
    contradiction is reported instead.

    """
    psy, loop, _ = shared_write_target
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(
            _coloured_inner(schedule), options={"atomics": True})
    assert "'atomics'" in str(err.value)
    assert "coloured" in str(err.value)


def test_lfric_kokkos_trans_refuses_a_disjoint_assertion_when_coloured(
        continuous_write_target):
    """Asserting disjointness on a coloured loop says two opposite things.

    Colouring is the other answer to cells that do share an element, so a
    loop that has been coloured and a caller saying nothing is shared cannot
    both be right about the same loop. Refused rather than resolved: which
    of the two the caller meant decides whether the uncoloured loop is safe.
    """
    psy, loop, _ = continuous_write_target
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans().validate(
            _coloured_inner(schedule), options={"disjoint_writes": True})

    assert ("'disjoint_writes' option is True on a coloured loop" in
            str(err.value))
    assert "capture the uncoloured loop" in str(err.value)


def test_lfric_kokkos_trans_atomics_off_is_redundant_when_coloured(
        shared_write_target):
    """Turning them off on a coloured loop asks for what it would do anyway.

    Stated rather than refused, because it contradicts nothing: the coloured
    arm generates no atomic whether or not the caller says so.
    """
    psy, loop, _ = shared_write_target
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)

    cpp = LFRicKokkosTrans().apply(
        _coloured_inner(schedule), options={"atomics": False})

    assert "atomic" not in cpp


def test_lfric_kokkos_trans_refuses_a_colouring_with_no_colour_count(
        shared_write_target, monkeypatch):
    """A colour map with no extent to address it by is refused, not guessed.

    LFRic creates the name for the number of colours where it first needs
    one, so a schedule can be coloured and not yet hold it. The generated
    View's first extent is the ``LayoutLeft`` stride, and a wrong stride
    reads a different cell for every colour but the first.
    """
    psy, loop, _ = shared_write_target
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)
    inner = _coloured_inner(schedule)
    kernel = inner.kernels()[0]
    # Asked of the rule rather than of a whole capture: LFRic reaches for the
    # same name to write the enclosing loop's own bound, so a schedule
    # missing it does not survive as far as apply().
    monkeypatch.setattr(LFRicKern, "ncolours_var",
                        property(lambda self: None))

    with pytest.raises(TransformationError) as err:
        LFRicKokkosTrans._colouring(
            kernel, inner, LFRicKokkosTrans._schedule(kernel), "cell")

    assert ("cannot capture a coloured loop whose invoke has no number of "
            "colours" in str(err.value))


def test_lfric_kokkos_trans_coloured_capture_looks_the_colourmap_up_once(
        shared_write_target):
    """The completion pass adds no look-up the set-up pass already made.

    Colouring before capturing means the set-up pass runs against a coloured
    Schedule and emits the colourmap look-ups itself. The completion pass
    exists for the other order and must find nothing to do here: emitting
    them again would assign the same pointer twice, in a preamble the
    coloured arm generates for every one of its captures.
    """
    psy, loop, _ = shared_write_target
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)

    LFRicKokkosTrans().apply(_coloured_inner(schedule))
    fortran = str(psy.gen)

    assert fortran.count("cmap => mesh%get_colour_map()") == 1
    assert fortran.count("ncolour = mesh%get_ncolours()") == 1


def test_lfric_kokkos_trans_captures_a_coloured_meshless_invoke(
        tmp_path, clear_module_manager_instance):
    # pylint: disable=unused-argument
    """An Invoke with no mesh of its own may be coloured and then captured.

    The mesh symbol a colouring creates is assigned by the set-up pass, from
    a kernel argument. Capturing first removes that kernel and leaves the
    assignment impossible, which is refused elsewhere; colouring first does
    not, because the set-up pass runs from the capture and the kernel is
    still in the tree when it does.
    """
    psy, loop, _ = _invoke(
        tmp_path, "inc_probe", _SHARED_WRITE_ALGORITHM, _SHARED_WRITE_KERNEL,
        dist_mem=False)
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)

    LFRicKokkosTrans().apply(_coloured_inner(schedule))
    fortran = str(psy.gen)

    assert "mesh => acc_proxy%vspace%get_mesh()" in fortran
    assert "cmap => mesh%get_colour_map()" in fortran
    assert "call inc_probe_kokkos(" in fortran


# The two capabilities above were written apart: one taught the launch to
# begin somewhere other than the first cell and to run over dofs, the other
# taught it to run one colour of a coloured loop. The kernel below is the one
# neither needed on its own -- a continuous-space kernel, so its loop can be
# coloured, holding a level loop, so its capture takes a team launch -- and
# the tests that follow it are about how the two answers compose.
_COLOURED_LEVEL_KERNEL = """
module column_scale_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_inc, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w2
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: column_scale_kernel_type
    type(arg_type) :: meta_args(2) = (/                              &
         arg_type(gh_field, gh_real, gh_inc,  w2),                   &
         arg_type(gh_field, gh_real, gh_read, w2) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: column_scale_code
  end type column_scale_kernel_type
contains
  subroutine column_scale_code(nlayers, field_out, field_in, &
                               ndf_w2, undf_w2, map_w2)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w2, undf_w2
    real(kind=r_def), dimension(undf_w2), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w2), intent(in) :: field_in
    integer(kind=i_def), dimension(ndf_w2), intent(in) :: map_w2
    integer(kind=i_def) :: k
    real(kind=r_def) :: scaling
    scaling = 0.5_r_def
    do k = 1, nlayers - 1
      field_out(map_w2(1) + k - 1) = scaling * field_in(map_w2(1) + k - 1)
    end do
    field_out(map_w2(1) + nlayers - 1) = field_in(map_w2(1) + nlayers - 1)
  end subroutine column_scale_code
end module column_scale_kernel_mod
"""


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_colours_a_team_launch(
        tmp_path, clear_module_manager_instance):
    """A coloured loop and a team launch are selected independently.

    Which launch shape a region takes is decided by what its kernel holds --
    a level loop here, so one team per cell -- and colouring decides what the
    launch's own index means. Nothing in either decision reads the other, so
    the two arrive together in one region: the team arithmetic names the
    index, the map turns it into a mesh cell, and the body indexes by the
    mesh cell as an uncoloured capture's does.
    """
    psy, loop, _ = _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _COLOURED_LEVEL_KERNEL)
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)

    cpp = LFRicKokkosTrans().apply(
        _coloured_inner(schedule), options={"team_size": 4})

    # The team shape, and the colour map inside it.
    assert "TeamPolicy(ncells, 4)," in cpp
    assert "const int cell_in_colour = team.league_rank();" in cpp
    assert "const int cell = cmap(colour - 1, cell_in_colour) - 1;" in cpp
    # The level loop is still spread over the team, and the statement after
    # it is still one member's.
    assert "Kokkos::TeamVectorRange(team, 1, (nlayers - 1) + 1)" in cpp
    assert "Kokkos::single(Kokkos::PerTeam(team)" in cpp
    # Colouring is the answer to this kernel's shared write, so there is no
    # atomic beside it, and the launch begins at the first cell of its colour.
    assert "Kokkos::atomic" not in cpp
    assert "first_cell" not in cpp

    # The PSy layer passes the map, the colour and this colour's cell count.
    call_line = [line for line in str(psy.gen).splitlines()
                 if "call column_scale_kokkos(" in line][0]
    assert "cmap, colour, ncolour, last_halo_cell_all_colours(colour,1)" \
        in call_line


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_a_coloured_loop_never_takes_a_first_cell(
        tmp_path, clear_module_manager_instance):
    """Colouring gives the loop it makes a lower bound of ``start``.

    This is why the two capabilities never meet in the one place they could
    contradict each other. A first cell counts the mesh's cells and a colour
    map's index counts one colour's, so a region carrying both would offset
    into the wrong sequence; the writer refuses the pair, and this is the
    front end's half of that answer -- it does not produce it. The halo-ness
    of the loop is not lost with the bound: it moves into the coloured
    loop's *upper* bound, which is why the capture is still of a loop that
    runs into the halo.
    """
    psy, loop, _ = _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _COLOURED_LEVEL_KERNEL)
    schedule = psy.invokes.invoke_list[0].schedule
    LFRicColourTrans().apply(loop)
    inner = _coloured_inner(schedule)

    # pylint: disable-next=protected-access
    assert inner._lower_bound_name == "start"
    assert inner.upper_bound_name == "colour_halo"
    assert LFRicKokkosTrans._start_name(inner) is None

    cpp = LFRicKokkosTrans().apply(inner)

    assert "first_cell" not in cpp


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_refuses_to_colour_a_halo_only_loop(
        tmp_path, clear_module_manager_instance):
    """The one loop that does take a first cell cannot be coloured at all.

    A loop over the halo alone is a loop over a discontinuous space -- that
    is what lets it skip the owned cells -- and ``LFRicColourTrans`` refuses
    those, so the second half of the pair the writer refuses is unreachable
    from LFRic as well as meaningless in the back end. Asserted rather than
    assumed, because the writer's refusal would otherwise be a rule with no
    stated reason for never firing.
    """
    _, loop, _ = _invoke(
        tmp_path, "halo_write", _HALO_CELL_ALGORITHM, _HALO_CELL_KERNEL)
    # pylint: disable-next=protected-access
    assert loop._lower_bound_name == "cell_halo_start"
    assert LFRicKokkosTrans._start_name(loop) == "first_cell"

    with pytest.raises(TransformationError) as error:
        LFRicColourTrans().apply(loop)

    assert ("Loops iterating over a discontinuous function space are not "
            "currently supported." in str(error.value))
