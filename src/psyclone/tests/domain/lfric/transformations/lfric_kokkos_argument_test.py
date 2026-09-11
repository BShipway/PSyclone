# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosArgumentMixin: the region's argument list."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _ALGORITHM, _KERNEL, _OPERATOR_ALGORITHM, _OPERATOR_KERNEL, _invoke)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans


# The production kernel with 'cell' declared as one of its own locals and
# genuinely used. GungHo has such kernels -- apply_helmholtz_operator_code
# counts a stencil branch with one -- and the name is the launch index's, so
# the generated declaration and the lambda parameter would share a C++ scope.
# That is a compile error rather than a wrong answer, which is why the
# transformation renames its index instead of leaving the collision to the
# compiler.
_CELL_LOCAL_KERNEL = _KERNEL.replace(
    "integer(kind=i_def) :: k, df",
    "integer(kind=i_def) :: k, df, cell").replace(
    "    do k = 0, nlayers - 1",
    "    cell = ndf_wtheta\n    do k = 0, nlayers - 1").replace(
    "      do df = 1, ndf_wtheta", "      do df = 1, cell")


_HALO_ALGORITHM = """
program kokkos_halo_test
  use field_mod, only : field_type
  use halo_read_kernel_mod, only : halo_read_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  call invoke(halo_read_kernel_type(out_field, in_field))
end program kokkos_halo_test
"""


# A kernel reading a field on a continuous space. Reading one is allowed --
# only writing one is refused -- and it is what makes distributed memory put a
# halo exchange in front of the loop, which none of the regions captured
# before this stage has. The stencil this stage's target uses puts one there
# too, for the same reason and by a different route.
_HALO_KERNEL = """
module halo_read_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w1, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: halo_read_kernel_type
    type(arg_type) :: meta_args(2) = (/                              &
         arg_type(gh_field, gh_real, gh_write, w3),                  &
         arg_type(gh_field, gh_real, gh_read,  w1) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: halo_read_code
  end type halo_read_kernel_type
contains
  subroutine halo_read_code(nlayers, field_out, field_in, &
                            ndf_w3, undf_w3, map_w3, &
                            ndf_w1, undf_w1, map_w1)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: ndf_w1, undf_w1
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w1), intent(in) :: field_in
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def), dimension(ndf_w1), intent(in) :: map_w1
    integer(kind=i_def) :: k
    do k = 0, nlayers - 1
      field_out(map_w3(1) + k) = field_in(map_w1(1) + k)
    end do
  end subroutine halo_read_code
end module halo_read_kernel_mod
"""


# apply_mixed_u_operator's shape, reduced to what makes it different: two
# operators over the same pair of spaces, each with an ncell of its own. The
# captured region this capability adds to the model has exactly this.
_TWO_OPERATOR_ALGORITHM = _OPERATOR_ALGORITHM.replace(
    "  type(operator_type) :: matrix",
    "  type(operator_type) :: matrix, matrix2").replace(
    "dg_matrix_vector_kernel_type(lhs, x, matrix)",
    "dg_matrix_vector_kernel_type(lhs, x, matrix, matrix2)")


# The fragments that variant swaps, held apart from the .replace() chain
# below rather than written into it: a Fortran continuation line indented by
# a Python expression as well as by itself does not fit in 79 columns.
_ONE_OPERATOR_ARG = """\
         arg_type(gh_operator, gh_real, gh_read,                           &
                               any_discontinuous_space_1, any_space_1) /)"""


_TWO_OPERATOR_ARGS = """\
         arg_type(gh_operator, gh_real, gh_read,                           &
                               any_discontinuous_space_1, any_space_1),    &
         arg_type(gh_operator, gh_real, gh_read,                           &
                               any_discontinuous_space_1, any_space_1) /)"""


_ONE_OPERATOR_SIGNATURE = """\
  subroutine dg_matrix_vector_code(cell, nlayers, lhs, x, ncell_3d, matrix, &
                                   ndf1, undf1, map1, ndf2, undf2, map2)"""


_TWO_OPERATOR_SIGNATURE = """\
  subroutine dg_matrix_vector_code(cell, nlayers, lhs, x,           &
                                   ncell_3d, matrix, ncell_3d_2, matrix2, &
                                   ndf1, undf1, map1, ndf2, undf2, map2)"""


_ONE_OPERATOR_DECL = (
    "    real(kind=r_def), dimension(ncell_3d,ndf1,ndf2), intent(in) "
    ":: matrix")


_TWO_OPERATOR_DECL = (
    _ONE_OPERATOR_DECL + "\n"
    "    real(kind=r_def), dimension(ncell_3d_2,ndf1,ndf2), intent(in) "
    ":: matrix2")


_TWO_OPERATOR_KERNEL = _OPERATOR_KERNEL.replace(
    "meta_args(3)", "meta_args(4)").replace(
    _ONE_OPERATOR_ARG, _TWO_OPERATOR_ARGS).replace(
    _ONE_OPERATOR_SIGNATURE, _TWO_OPERATOR_SIGNATURE).replace(
    "    integer(kind=i_def), intent(in) :: cell, nlayers, ncell_3d",
    "    integer(kind=i_def), intent(in) :: cell, nlayers, ncell_3d\n"
    "    integer(kind=i_def), intent(in) :: ncell_3d_2").replace(
    _ONE_OPERATOR_DECL, _TWO_OPERATOR_DECL).replace(
    "                      + matrix(ik:ik+nl, df, m) * x(i2:i2+nl)",
    "                      + (matrix(ik:ik+nl, df, m) &\n"
    "                      +  matrix2(ik:ik+nl, df, m)) * x(i2:i2+nl)")


@pytest.fixture(name="cell_local_target")
# pylint: disable-next=unused-argument
def cell_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares 'cell' as a local of its own."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _CELL_LOCAL_KERNEL)


@pytest.fixture(name="halo_target")
# pylint: disable-next=unused-argument
def halo_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose loop is preceded by a halo exchange."""
    return _invoke(
        tmp_path, "halo_read", _HALO_ALGORITHM, _HALO_KERNEL)


@pytest.fixture(name="two_operator_target")
# pylint: disable-next=unused-argument
def two_operator_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads two LMA operators."""
    return _invoke(
        tmp_path, "dg_matrix_vector", _TWO_OPERATOR_ALGORITHM,
        _TWO_OPERATOR_KERNEL)


def test_lfric_kokkos_trans_renames_the_index_a_kernel_declares(
        cell_local_target):
    """A kernel declaring 'cell' pushes the launch index off that name.

    The lambda parameter and the kernel's own declaration share one C++
    scope, so leaving both called 'cell' does not produce a subtly wrong
    answer -- it produces a translation unit the compiler rejects with
    'conflicting declaration'. The index is therefore named from the kernel's
    symbol table rather than fixed.
    """
    _, loop, _ = cell_local_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "KOKKOS_LAMBDA(const int cell_1)" in cpp
    assert "KOKKOS_LAMBDA(const int cell)" not in cpp
    # The kernel's own 'cell' keeps its name, and every sliced View follows
    # the launch index rather than the local.
    assert "int cell;" in cpp
    assert "map_wtheta((df - 1), cell_1)" in cpp
    assert "map_wtheta((df - 1), cell)" not in cpp


def test_lfric_kokkos_trans_keeps_a_preceding_halo_exchange(halo_target):
    """A halo exchange feeding the captured loop still resolves its depth.

    The exchange computes its depth by walking forward for the accesses that
    read its field, and those accesses live on the LFRic loop that ``apply``
    is about to replace with a plain ``Call``. Lowering the exchanges before
    the loop rather than after it is what keeps that walk able to find them;
    without it PSyclone raises ``InternalError`` from
    ``_compute_halo_read_info`` when the PSy layer is generated.
    """
    psy, loop, _ = halo_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert 'extern "C" void halo_read_kokkos(' in cpp
    assert "halo_exchange(depth=" in fortran
    assert "call halo_read_kokkos(" in fortran
    assert "call halo_read_code(" not in fortran
    # The exchange fills the halo the loop then reads, so it has to stay in
    # front of the launch rather than merely survive.
    assert fortran.index("halo_exchange(depth=") < \
        fortran.index("call halo_read_kokkos(")


def test_lfric_kokkos_trans_halo_exchanges_precede_the_region(
        halo_operator_target):
    """A region iterating into the halo does not exchange the halo itself.

    The exchange is the PSy layer's, before the launch, whether or not the
    launch reaches past the owned cells. Widening the bound moves where the
    region reads, not who fills what it reads: an exchange inside a region
    would be a communication call in device code.
    """
    psy, loop, _ = halo_operator_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert fortran.index("call weight_proxy%halo_exchange(depth=1)") < \
        fortran.index("call operator_setval_x_kokkos(")
    assert "halo_exchange" not in cpp
    assert "is_dirty" not in cpp
    assert "set_dirty" not in cpp


def test_lfric_kokkos_trans_lowers_no_exchange_outside_an_invoke(target):
    """A loop with no invoke schedule above it is left alone.

    The exchanges are reached through the loop's ``InvokeSchedule`` ancestor,
    and a detached loop has none. Returning rather than walking from ``None``
    is what lets the helper be called on a loop held outside the tree it was
    parsed into, as a unit test does.
    """
    _, loop, _ = target

    assert LFRicKokkosTrans._lower_halo_exchanges(loop.detach()) is None


def test_lfric_kokkos_trans_accepts_an_operator(operator_target):
    """An LMA operator becomes a rank-3 read-only View and nothing else.

    Every extent of the local stencil -- ncell_3d, ndf1 and ndf2 -- is itself
    a kernel formal, so the operator needs no machinery of its own: the
    existing View description covers it.
    """
    _, loop, _ = operator_target
    code = LFRicKokkosTrans().apply(loop)

    # The kernel only reads the operator, and its element type is const to
    # say so, but its role is readwrite: the role says where the storage is
    # and how long a copy of it stays good, not what this kernel does with
    # it. An operator is assembled by one kernel and applied by another, so
    # a copy taken on the first apply is wrong on the next timestep.
    assert ("auto matrix = lfric_kokkos::stage<\n"
            "      Kokkos::View<const double***, Kokkos::LayoutLeft, "
            "MemorySpace, ReadOnly>>(\n"
            "      matrix_data, lfric_kokkos::Role::readwrite, ncell_3d, "
            "ndf1, ndf2);") in code
    assert "const double *matrix_data" in code
    assert "const int ncell_3d" in code


def test_lfric_kokkos_trans_operator_dofmaps_keep_their_cell_index(
        operator_target):
    """Dropping the cell actual must not shift the formal/actual pairing.

    The per-cell detection walks the two lists together, so dropping the
    actual alone attributes every dofmap to the formal before it and the
    generated Views silently lose their cell dimension. That compiles and
    runs, and reads the wrong column.
    """
    _, loop, _ = operator_target
    code = LFRicKokkosTrans().apply(loop)

    for dofmap in ("map1", "map2"):
        assert (f"auto {dofmap} = lfric_kokkos::stage<\n"
                "      Kokkos::View<const int**, Kokkos::LayoutLeft, "
                "MemorySpace, ReadOnly>>(\n"
                f"      {dofmap}_data, lfric_kokkos::Role::readonly, "
                "ndf") in code
        role = "lfric_kokkos::Role::readonly"
        assert f"{dofmap}_data, {role}, ndf1, ncells)" in code or \
            f"{dofmap}_data, {role}, ndf2, ncells)" in code
        assert f"{dofmap}(" in code
    assert ", cell_1)" in code


def test_lfric_kokkos_trans_operator_call_drops_the_cell_actual(
        operator_target):
    """The generated call passes the operator, not the vanished counter."""
    psy, loop, _ = operator_target
    LFRicKokkosTrans().apply(loop)
    generated = str(psy.gen).lower()

    call = [line for line in generated.splitlines()
            if "call dg_matrix_vector_kokkos(" in line]
    assert call, generated
    arguments = call[0].split("(", 1)[1]
    assert "matrix_proxy%ncell_3d" in arguments
    assert "matrix_local_stencil" in arguments
    # The PSy loop counter is not passed, and having no other reader it is
    # dropped from the routine altogether.
    assert "cell," not in arguments
    assert ":: cell\n" not in generated


def test_lfric_kokkos_trans_accepts_two_operators(two_operator_target):
    """Two operators give two Views and still one cell declaration."""
    _, loop, _ = two_operator_target
    code = LFRicKokkosTrans().apply(loop)

    assert ("matrix_data, lfric_kokkos::Role::readwrite, "
            "ncell_3d, ndf1, ndf2);") in code
    assert ("matrix2_data, lfric_kokkos::Role::readwrite, "
            "ncell_3d_2, ndf1, ndf2);") in code
    assert code.count("const int cell = cell_1 + 1;") == 1


def test_lfric_kokkos_trans_places_a_local_array_in_scratch(local_target):
    """An automatic array becomes one scratch View the team shares.

    Two arrays over ``nlayers``, so the launch has to size both and the region
    has to build both. The sizing is what distinguishes scratch from a C++
    local: ``nlayers`` is a runtime value, so the bytes are asked for before
    the launch and the View is placed in them inside it.

    This kernel is the one that shows the two selections are independent. It
    has scratch *and* a level loop the dependence analysis accepts -- the
    write-back sweep, whose iterations touch disjoint elements -- so it takes
    the hierarchical launch, and the scratch it asked for is placed in team
    scratch rather than thread scratch. The flat launch is still reached, by
    a kernel whose every loop is a recurrence; see
    ``test_apply_keeps_tri_solve_flat``.
    """
    psy, loop, _ = local_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert 'extern "C" void column_solve_kokkos(' in cpp
    assert ("using ScratchSpace = "
            "Kokkos::DefaultExecutionSpace::scratch_memory_space;" in cpp)
    assert "partial_scratch_t::shmem_size(nlayers)" in cpp
    assert "swept_scratch_t::shmem_size(nlayers)" in cpp
    # Team scratch, not thread scratch: every member of the team works on the
    # one column, so one allocation is shared rather than one per member.
    assert "partial_scratch_t partial(team.team_scratch(0), nlayers);" in cpp
    assert "swept_scratch_t swept(team.team_scratch(0), nlayers);" in cpp
    assert "PerTeam(scratch_bytes)" in cpp

    # The launch is the hierarchical shape. It has no team-size probe and no
    # bounds guard, because the league is one team per cell rather than a
    # flat range of ranks that has to be folded onto cells.
    assert "TeamPolicy(ncells, Kokkos::AUTO)" in cpp
    assert "Kokkos::RangePolicy" not in cpp
    assert "team_size_recommended" not in cpp
    assert "if (cell >= ncells)" not in cpp

    # Neither local is declared in the body as well: a scratch View and a C++
    # array of the same name would not compile.
    assert "double partial[" not in cpp
    assert "double swept[" not in cpp

    # The two recurrences stay serial, each write made by one member and
    # published to the rest before the next statement reads it.
    assert ("Kokkos::single(Kokkos::PerTeam(team), [&]() {\n"
            "      partial((1 - 1)) = "
            "field_in((map_w3((1 - 1), cell) - 1));\n"
            "    });\n"
            "    team.team_barrier();" in cpp)
    assert "for(k=2; k<=nlayers; k+=1)" in cpp

    # The backward sweep counts down. Before the CWriter followed the step
    # sign this read 'k<=1' and ran no iterations, so the region built, linked
    # and returned the forward sweep's intermediates.
    assert "for(k=(nlayers - 1); k>=1; k+=(-1))" in cpp

    # The write-back is the loop that is spread. Its lambda parameter shadows
    # the region-scope 'k' the two serial sweeps drive, which is why that
    # declaration is emitted even though a parallel loop declares its own.
    assert "int k;" in cpp
    assert ("Kokkos::parallel_for(Kokkos::TeamVectorRange"
            "(team, 1, nlayers + 1),\n"
            "        [&](const int k) {" in cpp)

    # Nothing about the scratch reaches the Fortran side: it is allocated by
    # the launch, so the ABI is the same as any other region's.
    assert "subroutine column_solve_kokkos(" in fortran
    assert "partial" not in fortran
    assert "swept" not in fortran
    assert "call column_solve_code(" not in fortran


def test_lfric_kokkos_trans_declares_each_ndf_once(shared_write_target):
    """Asking which formals are shared declares nothing in the Invoke.

    The walk that answers it is an ArgOrdering, and an ArgOrdering writes
    the scalars it names into the Invoke's own symbol table whenever the
    kernel it is given is in a Schedule. Doing that here created
    ``ndf_<space>`` before LFRicFunctionSpaces did, whose declarations are
    made with ``new_symbol``: the space was then declared a second time as
    ``ndf_<space>_1`` and never assigned or read. LFRic builds the PSy layer
    with ``-Werror=unused-variable``, so the whole model stopped compiling.
    """
    psy, loop, _ = shared_write_target

    LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    for space in ("w2", "w3"):
        assert f"integer(kind=i_def) :: ndf_{space}\n" in fortran
        assert f"ndf_{space}_1" not in fortran
