# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the arguments an intergrid kernel's region takes."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _formals, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans


# Inter-grid kernels. The pair below has the shape of the model's
# sci_prolong_w0_kernel_mod and sci_restrict_w3_kernel_mod, and the argument
# order of PSyclone's own restrict_test_kernel_mod: the cell map and its three
# extents ahead of the fields, then each space's dofmap -- whole-mesh for the
# fine one, per-cell for the coarse. Both fields are on discontinuous spaces
# so that what the transformation is handed is a plain cell loop.
_PROLONG_ALGORITHM = """
program kokkos_prolong_test
  use field_mod, only : field_type
  use prolong_test_kernel_mod, only : prolong_test_kernel_type
  implicit none
  type(field_type) :: fine_field, coarse_field
  call invoke(prolong_test_kernel_type(fine_field, coarse_field))
end program kokkos_prolong_test
"""


_PROLONG_KERNEL = """
module prolong_test_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           gh_coarse, gh_fine, cell_column,                &
                           any_discontinuous_space_1,                      &
                           any_discontinuous_space_2
  use constants_mod, only : i_def, r_def
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: prolong_test_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_write, any_discontinuous_space_1,  &
                  mesh_arg=gh_fine),                                       &
         arg_type(gh_field, gh_real, gh_read, any_discontinuous_space_2,   &
                  mesh_arg=gh_coarse) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: prolong_test_code
  end type prolong_test_kernel_type
contains
  subroutine prolong_test_code(nlayers, cell_map, ncell_f_per_c_x,     &
                               ncell_f_per_c_y, ncell_f, fine_field,   &
                               coarse_field, ndf, undf_f, map_f,       &
                               undf_c, map_c)
    integer(kind=i_def), intent(in) :: nlayers, ncell_f_per_c_x
    integer(kind=i_def), intent(in) :: ncell_f_per_c_y, ncell_f
    integer(kind=i_def), intent(in) :: ndf, undf_f, undf_c
    integer(kind=i_def), dimension(ncell_f_per_c_x, ncell_f_per_c_y), &
        intent(in) :: cell_map
    real(kind=r_def), dimension(undf_f), intent(inout) :: fine_field
    real(kind=r_def), dimension(undf_c), intent(in) :: coarse_field
    integer(kind=i_def), dimension(ndf, ncell_f), intent(in) :: map_f
    integer(kind=i_def), dimension(ndf), intent(in) :: map_c
    integer(kind=i_def) :: k, df, x_idx, y_idx, fine_cell
    do y_idx = 1, ncell_f_per_c_y
      do x_idx = 1, ncell_f_per_c_x
        fine_cell = cell_map(x_idx, y_idx)
        do k = 0, nlayers - 1
          do df = 1, ndf
            fine_field(map_f(df, fine_cell) + k) = &
                coarse_field(map_c(df) + k)
          end do
        end do
      end do
    end do
  end subroutine prolong_test_code
end module prolong_test_kernel_mod
"""


_RESTRICT_ALGORITHM = """
program kokkos_restrict_test
  use field_mod, only : field_type
  use restrict_test_kernel_mod, only : restrict_test_kernel_type
  implicit none
  type(field_type) :: coarse_field, fine_field
  call invoke(restrict_test_kernel_type(coarse_field, fine_field))
end program kokkos_restrict_test
"""


_RESTRICT_KERNEL = """
module restrict_test_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_readwrite,      &
                           gh_read, gh_coarse, gh_fine, cell_column,       &
                           any_discontinuous_space_1,                      &
                           any_discontinuous_space_2
  use constants_mod, only : i_def, r_def
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: restrict_test_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_readwrite,                         &
                  any_discontinuous_space_1, mesh_arg=gh_coarse),          &
         arg_type(gh_field, gh_real, gh_read,                              &
                  any_discontinuous_space_2, mesh_arg=gh_fine) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: restrict_test_code
  end type restrict_test_kernel_type
contains
  subroutine restrict_test_code(nlayers, cell_map, ncell_f_per_c_x,      &
                                ncell_f_per_c_y, ncell_f, coarse_field,  &
                                fine_field, undf_c, map_c, ndf, undf_f,  &
                                map_f)
    integer(kind=i_def), intent(in) :: nlayers, ncell_f_per_c_x
    integer(kind=i_def), intent(in) :: ncell_f_per_c_y, ncell_f
    integer(kind=i_def), intent(in) :: ndf, undf_f, undf_c
    integer(kind=i_def), dimension(ncell_f_per_c_x, ncell_f_per_c_y), &
        intent(in) :: cell_map
    real(kind=r_def), dimension(undf_c), intent(inout) :: coarse_field
    real(kind=r_def), dimension(undf_f), intent(in) :: fine_field
    integer(kind=i_def), dimension(ndf), intent(in) :: map_c
    integer(kind=i_def), dimension(ndf, ncell_f), intent(in) :: map_f
    integer(kind=i_def) :: k, df, x_idx, y_idx, fine_cell
    do y_idx = 1, ncell_f_per_c_y
      do x_idx = 1, ncell_f_per_c_x
        fine_cell = cell_map(x_idx, y_idx)
        do k = 0, nlayers - 1
          do df = 1, ndf
            coarse_field(map_c(df) + k) = coarse_field(map_c(df) + k) &
                + fine_field(map_f(df, fine_cell) + k)
          end do
        end do
      end do
    end do
  end subroutine restrict_test_code
end module restrict_test_kernel_mod
"""


@pytest.fixture(name="prolongation_target")
# pylint: disable-next=unused-argument
def prolongation_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke of an inter-grid kernel writing the fine field."""
    return _invoke(
        tmp_path, "prolong_test", _PROLONG_ALGORITHM, _PROLONG_KERNEL)


@pytest.fixture(name="restriction_target")
# pylint: disable-next=unused-argument
def restriction_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke of an inter-grid kernel writing the coarse field."""
    return _invoke(
        tmp_path, "restrict_test", _RESTRICT_ALGORITHM, _RESTRICT_KERNEL)


def test_lfric_kokkos_trans_accepts_a_prolongation(prolongation_target):
    """An inter-grid kernel puts both meshes' arguments on the ABI.

    The region knows one loop but two meshes: the cell map and its three
    extents describe the fine cells a coarse cell covers, and the fine space
    brings a whole-mesh dofmap where the coarse space brings a per-cell one.
    They are asserted as an ordered sequence rather than one by one, because
    the region's formals are positional and an ABI that carries every name in
    the wrong order is wrong in the way that compiles.
    """
    _, loop, kernel = prolongation_target
    assert kernel.is_intergrid
    code = LFRicKokkosTrans().apply(loop)

    intergrid = ["cell_map_data", "ncell_f_per_c_x", "ncell_f_per_c_y",
                 "ncell_f", "map_f_data", "map_c_data"]
    assert [formal for formal in _formals(code) if formal in intergrid] == \
        intergrid
    # The fine dofmap is indexed by a cell the map chooses, so it arrives
    # whole; the coarse one is the launch's own cell and is sliced as any
    # single-mesh dofmap is.
    assert "Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> map_f(map_f_data, ndf, ncell_f);" in code
    assert "Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> map_c(map_c_data, ndf, ncells);" in code
    assert "map_f((df - 1), (fine_cell - 1))" in code
    assert "map_c((df - 1), cell)" in code


def test_lfric_kokkos_trans_intergrid_launches_over_coarse_cells(
        prolongation_target):
    """The launch runs over the coarse mesh, and the fine count is data.

    A prolongation visits every coarse cell once and writes the fine cells
    beneath it. Launching over the fine mesh instead would run the kernel
    body once per fine cell -- more iterations than there are coarse cells to
    read, each of them writing the whole fan-out again. Both meshes' counts
    are on the ABI, so which one bounds the policy is asserted from the
    Fortran actual as well as from the C++.
    """
    psy, loop, _ = prolongation_target
    code = LFRicKokkosTrans().apply(loop)
    generated = str(psy.gen).lower()

    assert "Kokkos::RangePolicy<>(0, ncells)" in code
    assert "loop0_stop = mesh_coarse_field%get_last_edge_cell()" in generated

    call = [line for line in generated.splitlines()
            if "call prolong_test_kokkos(" in line]
    assert call, generated
    actuals = [actual.strip() for actual in
               call[0].split("(", 1)[1].rsplit(")", 1)[0].split(",")]
    # 'ncells' is the last formal, and the coarse bound is what fills it.
    assert actuals[-1] == "loop0_stop"
    # The fine mesh's cell count is passed too, but as an extent rather than
    # as the bound.
    assert "ncell_fine_field" in actuals
    assert actuals.index("ncell_fine_field") != len(actuals) - 1


def test_lfric_kokkos_trans_accepts_a_restriction(restriction_target):
    """A restriction is the same ABI with the coarse field written.

    Which mesh is written does not change what the region takes, only which
    View loses its ``const``. The order does change: the coarse field is the
    first argument here, so its dofmap precedes the fine mesh's.
    """
    _, loop, kernel = restriction_target
    assert kernel.is_intergrid
    code = LFRicKokkosTrans().apply(loop)

    intergrid = ["cell_map_data", "ncell_f_per_c_x", "ncell_f_per_c_y",
                 "ncell_f", "map_c_data", "map_f_data"]
    assert [formal for formal in _formals(code) if formal in intergrid] == \
        intergrid
    assert "Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace, " \
        "Unmanaged> coarse_field(coarse_field_data, undf_c);" in code
    assert "Kokkos::View<const double*, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> fine_field(fine_field_data, undf_f);" in code
    assert "double *coarse_field_data" in code
    assert "const double *fine_field_data" in code


def test_lfric_kokkos_trans_intergrid_cell_map_indexing(prolongation_target):
    """The cell map keeps the kernel's subscript order with cell appended.

    The Fortran kernel declares ``cell_map(ncell_f_per_c_x,
    ncell_f_per_c_y)`` and reads ``cell_map(x_idx, y_idx)``; the PSy layer
    hands the region the whole rank-3 array, one plane per coarse cell.
    Transposing the two horizontal extents would still compile and would
    still be in bounds on a square fan-out, returning a different fine cell,
    so the extents and the subscripts are asserted together.
    """
    _, loop, _ = prolongation_target
    code = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::View<const int***, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> cell_map(cell_map_data, ncell_f_per_c_x, " \
        "ncell_f_per_c_y, ncells);" in code
    assert "fine_cell = cell_map((x_idx - 1), (y_idx - 1), cell);" in code
