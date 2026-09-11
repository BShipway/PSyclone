# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the stencil arguments a region takes."""

# pylint: disable=protected-access

import re

import pytest

from lfric_kokkos_sources import _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans


_STENCIL_ALGORITHM = """
program kokkos_stencil_test
  use constants_mod, only : i_def
  use field_mod, only : field_type
  use stencil_sum_kernel_mod, only : stencil_sum_kernel_type
  implicit none
  type(field_type) :: field_out, field_in
  integer(kind=i_def) :: extent = 1
  call invoke(stencil_sum_kernel_type(field_out, field_in, extent))
end program kokkos_stencil_test
"""


# apply_helmholtz_operator_code's stencil access with its algebra removed: a
# CROSS2D branch loop over a sliced dofmap, bounded by a sliced size array.
# The three formals the stencil adds are all used, because the point of the
# fixture is what 'apply' does with each of them -- 'smap_sizes' and 'smap'
# become Views with the cell index appended, 'max_length' stays a scalar --
# and an unused formal would still be declared but would not be indexed.
#
# The branch counter is 'step' rather than the production kernel's 'cell'.
# That collision is real and is Task 5.3's subject, tested by
# '_CELL_LOCAL_KERNEL'; repeating it here would make a stencil failure and a
# naming failure indistinguishable.
_STENCIL_KERNEL = """
module stencil_sum_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column, stencil, cross2d
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: stencil_sum_kernel_type
    type(arg_type) :: meta_args(2) = (/                                 &
         arg_type(gh_field, gh_real, gh_write, w3),                     &
         arg_type(gh_field, gh_real, gh_read,  w3, stencil(cross2d)) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: stencil_sum_code
  end type stencil_sum_kernel_type
contains
  subroutine stencil_sum_code(nlayers, field_out, field_in, &
                              smap_sizes, max_length, smap, &
                              ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: max_length
    integer(kind=i_def), dimension(4), intent(in) :: smap_sizes
    integer(kind=i_def), dimension(ndf_w3, max_length, 4), intent(in) :: smap
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df, branch, step
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        field_out(map_w3(df) + k) = 0.0_r_def
        do branch = 1, 4
          do step = 1, smap_sizes(branch)
            field_out(map_w3(df) + k) = field_out(map_w3(df) + k) + &
                field_in(smap(df, step, branch) + k)
          end do
        end do
      end do
    end do
  end subroutine stencil_sum_code
end module stencil_sum_kernel_mod
"""


_STENCIL_1D_ALGORITHM = _STENCIL_ALGORITHM.replace(
    "stencil_sum", "stencil_line").replace(
    "kokkos_stencil_test", "kokkos_stencil_line_test")


# The same kernel through a 1-D CROSS stencil. LFRic gives a 1-D stencil's
# size to the kernel as a *scalar* formal, fed per cell from
# 'field_in_stencil_size(cell)', where CROSS2D gives an array formal fed from
# a whole array. That scalar is what the per-cell View kind exists for: the
# region takes 'field_in_stencil_size' whole and subscripts it by the cell the
# thread is on. The dofmap loses its branch dimension and gains the per-cell
# size as its declared second extent, which is the other half of the same
# shape and is why the region also carries that dofmap's storage extent.
_STENCIL_1D_KERNEL = """
module stencil_line_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column, stencil, cross
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: stencil_line_kernel_type
    type(arg_type) :: meta_args(2) = (/                               &
         arg_type(gh_field, gh_real, gh_write, w3),                   &
         arg_type(gh_field, gh_real, gh_read,  w3, stencil(cross)) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: stencil_line_code
  end type stencil_line_kernel_type
contains
  subroutine stencil_line_code(nlayers, field_out, field_in, &
                               smap_size, smap, &
                               ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: smap_size
    integer(kind=i_def), dimension(ndf_w3, smap_size), intent(in) :: smap
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df, step
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        field_out(map_w3(df) + k) = 0.0_r_def
        do step = 1, smap_size
          field_out(map_w3(df) + k) = field_out(map_w3(df) + k) + &
              field_in(smap(df, step) + k)
        end do
      end do
    end do
  end subroutine stencil_line_code
end module stencil_line_kernel_mod
"""


_STENCIL_REGION_ALGORITHM = _STENCIL_ALGORITHM.replace(
    "stencil_sum", "stencil_region").replace(
    "kokkos_stencil_test", "kokkos_stencil_region_test")


# The same kernel again through a REGION stencil, whose PSy-layer arguments
# have exactly the 1-D shape -- 'field_in_stencil_size(cell)' beside
# 'field_in_stencil_dofmap(:,:,cell)' -- and differ from CROSS only in which
# cells the dofmap names. Written out through a substitution rather than
# patched into the 1-D kernel's metadata, so that the shape reaching the
# transformation is the parser's reading of 'stencil(region)' rather than the
# test's assertion about it.
_STENCIL_REGION_KERNEL = _STENCIL_1D_KERNEL.replace(
    "stencil_line", "stencil_region").replace("cross", "region")


@pytest.fixture(name="stencil_target")
# pylint: disable-next=unused-argument
def stencil_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads through a CROSS2D stencil."""
    return _invoke(
        tmp_path, "stencil_sum", _STENCIL_ALGORITHM, _STENCIL_KERNEL)


@pytest.fixture(name="stencil_1d_target")
# pylint: disable-next=unused-argument
def stencil_1d_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads through a 1-D CROSS stencil."""
    return _invoke(
        tmp_path, "stencil_line", _STENCIL_1D_ALGORITHM, _STENCIL_1D_KERNEL)


@pytest.fixture(name="stencil_region_target")
# pylint: disable-next=unused-argument
def stencil_region_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads through a REGION stencil."""
    return _invoke(
        tmp_path, "stencil_region", _STENCIL_REGION_ALGORITHM,
        _STENCIL_REGION_KERNEL)


def test_lfric_kokkos_trans_accepts_a_cross2d_stencil(stencil_target):
    """A CROSS2D stencil needs no argument machinery of its own.

    Its three PSy-layer actuals are a cell-sliced rank-2 size array, a plain
    integer and a cell-sliced rank-4 dofmap, and 'apply''s existing per-cell
    rule already passes each of them correctly: the two slices are
    ArrayReferences, so they are passed whole with the cell index appended,
    exactly as 'map_w3(:,cell)' already is, and 'max_length' is a scalar that
    stays one. The test asserts the shapes rather than the mere absence of a
    refusal, because it is the shapes that carry that claim.
    """
    psy, loop, _ = stencil_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert (("auto smap_sizes = lfric_kokkos::stage<\n"
             "      Kokkos::View<const int**, Kokkos::LayoutLeft, "
             "MemorySpace, ReadOnly>>(\n"
             "      smap_sizes_data, lfric_kokkos::Role::readonly, 4, "
             "ncells);") in cpp)
    assert (("auto smap = lfric_kokkos::stage<\n"
             "      Kokkos::View<const int****, Kokkos::LayoutLeft, "
             "MemorySpace, ReadOnly>>(\n"
             "      smap_data, lfric_kokkos::Role::readonly, ndf_w3, "
             "max_length, 4, ncells);") in cpp)
    assert "const int max_length" in cpp
    assert "max_length_data" not in cpp
    assert "smap((df - 1), (step - 1), (branch - 1), cell)" in cpp
    assert "smap_sizes((branch - 1), cell)" in cpp

    # The three actuals are passed whole, the cell index having moved into the
    # Views' last extent, and the halo exchange the stencil puts in front of
    # the loop is still there and still ahead of the launch.
    fortran = str(psy.gen)
    assert "integer(c_int), dimension(*), intent(in) :: smap_sizes" in fortran
    assert "integer(c_int), value :: max_length" in fortran
    assert "integer(c_int), dimension(*), intent(in) :: smap" in fortran
    assert ("call stencil_sum_kokkos(nlayers_field_out, field_out_data, "
            "field_in_data, field_in_stencil_size, "
            "field_in_max_branch_length, field_in_stencil_dofmap, ndf_w3, "
            "undf_w3, map_w3, loop0_stop)" in fortran)
    assert fortran.index("halo_exchange(depth=extent)") < fortran.index(
        "call stencil_sum_kokkos(")


def test_lfric_kokkos_trans_stencil_size_is_indexed_by_cell(
        stencil_region_target):
    """A stencil's size is one number per cell, not one number.

    The PSy layer evaluates 'field_in_stencil_size(cell)' inside the cell
    loop it is about to lose, so the value it would hand a region is whichever
    cell the Fortran loop happened to be on. A region runs every cell at once,
    and the sizes differ wherever the mesh is not uniform, so the size crosses
    the ABI as the whole rank-1 array and the region subscripts it by the cell
    the thread is on -- the same change of shape a dofmap already gets.

    The assertion is over *every* appearance of the name rather than over one
    of them, because a size read correctly in one place and wrongly in another
    compiles, links, runs and reads another cell's stencil.
    """
    psy, loop, _ = stencil_region_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert (("auto smap_size = lfric_kokkos::stage<\n"
             "      Kokkos::View<const int*, Kokkos::LayoutLeft, "
             "MemorySpace, ReadOnly>>(\n"
             "      smap_size_data, lfric_kokkos::Role::readonly, "
             "ncells);") in cpp)
    assert "const int *smap_size_data" in cpp
    # '\b' stops either pattern reaching 'smap_size_data' or 'smap_size_max',
    # so what is left is the size itself: never bare, and subscripted by the
    # cell everywhere but the View construction.
    body = cpp.replace("auto smap_size = ", "")
    assert not re.findall(r"\bsmap_size\b(?!\()", body)
    assert set(re.findall(r"\bsmap_size\(([^)]*)\)", cpp)) == {"cell"}

    fortran = str(psy.gen)
    assert "integer(c_int), dimension(*), intent(in) :: smap_size" in fortran
    assert "field_in_stencil_size," in fortran


def test_lfric_kokkos_trans_accepts_a_region_stencil(stencil_region_target):
    """A REGION stencil is named in the accepted set and is captured.

    Its dofmap is the 1-D shape -- 'field_in_stencil_dofmap(:,:,cell)' -- so
    the region takes it whole as a rank-3 View. Its second extent is the one
    the kernel declares, which is the *per-cell* size, and a View cannot be
    strided by a value that varies cell to cell: the region carries the
    dofmap's storage extent as a scalar of its own and sizes the View from
    that instead.
    """
    psy, loop, _ = stencil_region_target

    assert "region" in LFRicKokkosTrans._SUPPORTED_STENCILS

    cpp = LFRicKokkosTrans().apply(loop)

    assert (("auto smap = lfric_kokkos::stage<\n"
             "      Kokkos::View<const int***, Kokkos::LayoutLeft, "
             "MemorySpace, ReadOnly>>(\n"
             "      smap_data, lfric_kokkos::Role::readonly, ndf_w3, "
             "smap_size_max, ncells);")
            in cpp)
    assert "const int smap_size_max" in cpp
    assert "smap((df - 1), (step - 1), cell)" in cpp

    fortran = str(psy.gen)
    assert "STENCIL_REGION" in fortran
    assert "integer(c_int), value :: smap_size_max" in fortran
    assert ("call stencil_region_kokkos(nlayers_field_out, field_out_data, "
            "field_in_data, field_in_stencil_size, field_in_stencil_dofmap, "
            "ndf_w3, undf_w3, map_w3, SIZE(field_in_stencil_dofmap, dim=2), "
            "loop0_stop)" in fortran)


def test_lfric_kokkos_trans_accepts_a_cross_stencil_with_a_variable_extent(
        stencil_1d_target):
    """A 1-D CROSS stencil is captured, and its depth need not be a literal.

    The algorithm layer supplies 'extent' as a variable, so the PSy layer
    builds the dofmap at a depth it does not know until it runs and the halo
    exchange in front of the loop is written to that same variable. Nothing
    the region carries may therefore assume a size: the stencil size is the
    whole per-cell array and the dofmap's storage extent is measured from the
    array the PSy layer actually built.
    """
    psy, loop, _ = stencil_1d_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert (("auto smap_size = lfric_kokkos::stage<\n"
             "      Kokkos::View<const int*, Kokkos::LayoutLeft, "
             "MemorySpace, ReadOnly>>(\n"
             "      smap_size_data, lfric_kokkos::Role::readonly, "
             "ncells);") in cpp)
    assert (("auto smap = lfric_kokkos::stage<\n"
             "      Kokkos::View<const int***, Kokkos::LayoutLeft, "
             "MemorySpace, ReadOnly>>(\n"
             "      smap_data, lfric_kokkos::Role::readonly, ndf_w3, "
             "smap_size_max, ncells);")
            in cpp)
    assert "for(step=1; step<=smap_size(cell); step+=1)" in cpp

    fortran = str(psy.gen)
    assert "STENCIL_CROSS, extent" in fortran
    assert fortran.index("halo_exchange(depth=extent)") < fortran.index(
        "call stencil_line_kokkos(")
