# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the basis and evaluator arguments a region takes."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _FACE_QUADRATURE_ALGORITHM, _FACE_QUADRATURE_KERNEL, _invoke)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans


# Kernels asking for basis data. It arrives as arguments the PSy layer has
# computed before the loop, so these are written the way the model writes
# them -- explicit shapes, every extent a formal of the same kernel -- rather
# than with the assumed-shape declarations PSyclone's own test kernels use,
# which the region contract refuses for reasons that have nothing to do with
# quadrature.
_QUADRATURE_ALGORITHM = """
program kokkos_quadrature_test
  use field_mod, only : field_type
  use quadrature_xyoz_mod, only : quadrature_xyoz_type
  use qr_weight_kernel_mod, only : qr_weight_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  type(quadrature_xyoz_type) :: qr
  call invoke(qr_weight_kernel_type(out_field, in_field, qr))
end program kokkos_quadrature_test
"""


_QUADRATURE_KERNEL = """
module qr_weight_kernel_mod
  use argument_mod, only : arg_type, func_type, gh_field, gh_real, gh_write, &
                           gh_read, gh_basis, cell_column, gh_quadrature_XYoZ
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: qr_weight_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_write, w3),                        &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    type(func_type) :: meta_funcs(1) = (/                                  &
         func_type(w3, gh_basis) /)
    integer :: gh_shape = gh_quadrature_XYoZ
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: qr_weight_code
  end type qr_weight_kernel_type
contains
  subroutine qr_weight_code(nlayers, field_out, field_in,                  &
                            ndf_w3, undf_w3, map_w3, basis_w3,             &
                            np_xy, np_z, weights_xy, weights_z)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: np_xy, np_z
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    real(kind=r_def), dimension(np_xy), intent(in) :: weights_xy
    real(kind=r_def), dimension(np_z), intent(in) :: weights_z
    real(kind=r_def), dimension(1,ndf_w3,np_xy,np_z), intent(in) :: basis_w3
    integer(kind=i_def) :: k, df, qp1, qp2
    real(kind=r_def) :: total
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        total = 0.0_r_def
        do qp2 = 1, np_z
          do qp1 = 1, np_xy
            total = total + weights_xy(qp1) * weights_z(qp2)               &
                  * basis_w3(1,df,qp1,qp2)
          end do
        end do
        field_out(map_w3(df) + k) = total * field_in(map_w3(df) + k)
      end do
    end do
  end subroutine qr_weight_code
end module qr_weight_kernel_mod
"""


# The same kernel asking for the derivative as well. A basis on W3 has one
# component and its derivative three, so the pair is the case that shows the
# leading extent is read per array rather than assumed once. Written out
# rather than derived from the kernel above with .replace(): the fragments
# that would have to be matched are Fortran continuation lines, which do not
# fit in 79 columns once they are indented as Python arguments as well.
_DIFF_BASIS_KERNEL = """
module qr_weight_kernel_mod
  use argument_mod, only : arg_type, func_type, gh_field, gh_real, gh_write, &
                           gh_read, gh_basis, gh_diff_basis, cell_column,    &
                           gh_quadrature_XYoZ
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: qr_weight_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_write, w3),                        &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    type(func_type) :: meta_funcs(1) = (/                                  &
         func_type(w3, gh_basis, gh_diff_basis) /)
    integer :: gh_shape = gh_quadrature_XYoZ
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: qr_weight_code
  end type qr_weight_kernel_type
contains
  subroutine qr_weight_code(nlayers, field_out, field_in,                  &
                            ndf_w3, undf_w3, map_w3, basis_w3,             &
                            diff_basis_w3, np_xy, np_z, weights_xy,        &
                            weights_z)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: np_xy, np_z
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    real(kind=r_def), dimension(np_xy), intent(in) :: weights_xy
    real(kind=r_def), dimension(np_z), intent(in) :: weights_z
    real(kind=r_def), dimension(1,ndf_w3,np_xy,np_z), intent(in) :: basis_w3
    real(kind=r_def), dimension(3,ndf_w3,np_xy,np_z), intent(in) ::        &
                                                            diff_basis_w3
    integer(kind=i_def) :: k, df, qp1, qp2
    real(kind=r_def) :: total
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        total = 0.0_r_def
        do qp2 = 1, np_z
          do qp1 = 1, np_xy
            total = total + weights_xy(qp1) * weights_z(qp2)               &
                  * (basis_w3(1,df,qp1,qp2)                                &
                  +  diff_basis_w3(3,df,qp1,qp2))
          end do
        end do
        field_out(map_w3(df) + k) = total * field_in(map_w3(df) + k)
      end do
    end do
  end subroutine qr_weight_code
end module qr_weight_kernel_mod
"""


_EVALUATOR_ALGORITHM = """
program kokkos_evaluator_test
  use field_mod, only : field_type
  use eval_project_kernel_mod, only : eval_project_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  call invoke(eval_project_kernel_type(out_field, in_field))
end program kokkos_evaluator_test
"""


# An evaluator carries no weights and no point counts: the basis is tabulated
# at the nodal points of the target space, which is the space of the written
# field, so its last extent is that space's ndf.
_EVALUATOR_KERNEL = """
module eval_project_kernel_mod
  use argument_mod, only : arg_type, func_type, gh_field, gh_real, gh_write, &
                           gh_read, gh_basis, gh_diff_basis, cell_column,    &
                           gh_evaluator
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w1, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: eval_project_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_write, w3),                        &
         arg_type(gh_field, gh_real, gh_read,  w1) /)
    type(func_type) :: meta_funcs(2) = (/                                  &
         func_type(w3, gh_basis),                                          &
         func_type(w1, gh_diff_basis) /)
    integer :: gh_shape = gh_evaluator
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: eval_project_code
  end type eval_project_kernel_type
contains
  subroutine eval_project_code(nlayers, field_out, field_in,               &
                               ndf_w3, undf_w3, map_w3, basis_w3_on_w3,    &
                               ndf_w1, undf_w1, map_w1, diff_basis_w1_on_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: ndf_w1, undf_w1
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def), dimension(ndf_w1), intent(in) :: map_w1
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w1), intent(in) :: field_in
    real(kind=r_def), dimension(1,ndf_w3,ndf_w3), intent(in) ::            &
                                                           basis_w3_on_w3
    real(kind=r_def), dimension(3,ndf_w1,ndf_w3), intent(in) ::            &
                                                      diff_basis_w1_on_w3
    integer(kind=i_def) :: k, df, dg
    real(kind=r_def) :: total
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        total = 0.0_r_def
        do dg = 1, ndf_w1
          total = total + basis_w3_on_w3(1,df,df)                          &
                * diff_basis_w1_on_w3(3,dg,df) * field_in(map_w1(dg) + k)
        end do
        field_out(map_w3(df) + k) = total
      end do
    end do
  end subroutine eval_project_code
end module eval_project_kernel_mod
"""


_BOTH_SHAPES_ALGORITHM = """
program kokkos_both_shapes_test
  use field_mod, only : field_type
  use quadrature_xyoz_mod, only : quadrature_xyoz_type
  use both_shapes_kernel_mod, only : both_shapes_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  type(quadrature_xyoz_type) :: qr
  call invoke(both_shapes_kernel_type(out_field, in_field, qr))
end program kokkos_both_shapes_test
"""


# A kernel may ask for both shapes at once, in which case every function space
# it names carries two basis arrays: one over the quadrature points and one
# over the target space's nodes.
_BOTH_SHAPES_KERNEL = """
module both_shapes_kernel_mod
  use argument_mod, only : arg_type, func_type, gh_field, gh_real, gh_write, &
                           gh_read, gh_basis, cell_column,                   &
                           gh_quadrature_XYoZ, gh_evaluator
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w1, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: both_shapes_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_write, w3),                        &
         arg_type(gh_field, gh_real, gh_read,  w1) /)
    type(func_type) :: meta_funcs(2) = (/                                  &
         func_type(w3, gh_basis),                                          &
         func_type(w1, gh_basis) /)
    integer :: gh_shape(2) = (/ gh_quadrature_XYoZ, gh_evaluator /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: both_shapes_code
  end type both_shapes_kernel_type
contains
  subroutine both_shapes_code(nlayers, field_out, field_in, ndf_w3, undf_w3, &
                              map_w3, basis_w3_qr, basis_w3_on_w3,          &
                              ndf_w1, undf_w1, map_w1, basis_w1_qr,         &
                              basis_w1_on_w3, np_xy, np_z, weights_xy,      &
                              weights_z)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: ndf_w1, undf_w1, np_xy, np_z
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def), dimension(ndf_w1), intent(in) :: map_w1
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w1), intent(in) :: field_in
    real(kind=r_def), dimension(1,ndf_w3,np_xy,np_z), intent(in) ::        &
                                                              basis_w3_qr
    real(kind=r_def), dimension(1,ndf_w3,ndf_w3), intent(in) ::            &
                                                           basis_w3_on_w3
    real(kind=r_def), dimension(3,ndf_w1,np_xy,np_z), intent(in) ::        &
                                                              basis_w1_qr
    real(kind=r_def), dimension(3,ndf_w1,ndf_w3), intent(in) ::            &
                                                           basis_w1_on_w3
    real(kind=r_def), dimension(np_xy), intent(in) :: weights_xy
    real(kind=r_def), dimension(np_z), intent(in) :: weights_z
    integer(kind=i_def) :: k, df, qp1, qp2
    real(kind=r_def) :: total
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        total = 0.0_r_def
        do qp2 = 1, np_z
          do qp1 = 1, np_xy
            total = total + weights_xy(qp1) * weights_z(qp2)               &
                  * basis_w3_qr(1,df,qp1,qp2) * basis_w1_qr(3,1,qp1,qp2)   &
                  * field_in(map_w1(1) + k)
          end do
        end do
        total = total + basis_w3_on_w3(1,df,df) * basis_w1_on_w3(3,1,df)
        field_out(map_w3(df) + k) = total
      end do
    end do
  end subroutine both_shapes_code
end module both_shapes_kernel_mod
"""


@pytest.fixture(name="quadrature_target")
# pylint: disable-next=unused-argument
def quadrature_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads XYoZ quadrature basis data."""
    return _invoke(
        tmp_path, "qr_weight", _QUADRATURE_ALGORITHM, _QUADRATURE_KERNEL)


@pytest.fixture(name="diff_basis_target")
# pylint: disable-next=unused-argument
def diff_basis_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads a basis and its derivative."""
    return _invoke(
        tmp_path, "qr_weight", _QUADRATURE_ALGORITHM, _DIFF_BASIS_KERNEL)


@pytest.fixture(name="evaluator_target")
# pylint: disable-next=unused-argument
def evaluator_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads evaluator basis data."""
    return _invoke(
        tmp_path, "eval_project", _EVALUATOR_ALGORITHM, _EVALUATOR_KERNEL)


@pytest.fixture(name="both_shapes_target")
# pylint: disable-next=unused-argument
def both_shapes_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel asks for quadrature and an evaluator."""
    return _invoke(
        tmp_path, "both_shapes", _BOTH_SHAPES_ALGORITHM, _BOTH_SHAPES_KERNEL)


def test_lfric_kokkos_trans_accepts_a_quadrature_kernel(quadrature_target):
    """XYoZ quadrature crosses the ABI as data the PSy layer already holds.

    The rule arrives as four formals -- two point counts by value and two
    weight arrays as read-only Views -- and each function space asking for a
    basis adds one more View, shaped (dim, ndf, np_xy, np_z). Every extent is
    a formal of the same kernel, so nothing needs to be computed inside the
    region; ``ArgOrdering`` puts the quadrature rule last, after the
    per-function-space arguments, and the generated signature follows it.
    """
    _, loop, _ = quadrature_target
    code = LFRicKokkosTrans().apply(loop)

    assert "const int np_xy" in code
    assert "const int np_z" in code
    assert "Kokkos::View<const double*, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> weights_xy(weights_xy_data, np_xy);" in code
    assert "Kokkos::View<const double*, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> weights_z(weights_z_data, np_z);" in code
    assert "Kokkos::View<const double****, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> basis_w3(basis_w3_data, 1, ndf_w3, np_xy, np_z);" in code

    signature = code.split(") {\n")[0]
    parameters = [
        parameter.strip().split()[-1].lstrip("*")
        for parameter in signature.split(",") if parameter.strip().split()]
    assert parameters.index("basis_w3_data") < parameters.index("np_xy")
    assert parameters.index("np_z") < parameters.index("weights_xy_data")


def test_lfric_kokkos_trans_quadrature_call_passes_the_psy_arrays(
        quadrature_target):
    """The PSy layer passes the arrays it computed, not a slice of them.

    The basis array is allocated and filled by ``compute_function`` before the
    loop and is whole-array data, unlike a dofmap, which the PSy layer slices
    by cell. Passing it sliced would give the region a cell dimension it does
    not have and read past the end of the allocation.
    """
    psy, loop, _ = quadrature_target
    LFRicKokkosTrans().apply(loop)
    generated = str(psy.gen).lower()

    call = [line for line in generated.splitlines()
            if "call qr_weight_kokkos(" in line]
    assert call, generated
    arguments = call[0].split("(", 1)[1]
    for actual in ("basis_w3_qr", "np_xy_qr", "np_z_qr", "weights_xy_qr",
                   "weights_z_qr"):
        assert actual in arguments
    assert "basis_w3_qr(" not in arguments
    assert "np_xy_qr = qr_proxy%np_xy" in generated
    assert "weights_xy_qr => qr_proxy%weights_xy" in generated


def test_lfric_kokkos_trans_accepts_an_evaluator_kernel(evaluator_target):
    """An evaluator tabulates the basis at a target space's nodal points.

    There is no quadrature rule, so no weights and no point counts cross the
    ABI at all. The basis is rank 3 -- (dim, ndf, ndf of the target space) --
    and the target is the space of the written field, W3 here, which is why
    the W1 derivative is shaped by ``ndf_w3`` and not by its own ``ndf_w1``.
    """
    _, loop, kernel = evaluator_target
    assert kernel.eval_shapes == ["gh_evaluator"]
    assert not kernel.qr_required
    code = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::View<const double***, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> basis_w3_on_w3(basis_w3_on_w3_data, 1, ndf_w3, ndf_w3);" \
        in code
    assert "Kokkos::View<const double***, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> diff_basis_w1_on_w3(diff_basis_w1_on_w3_data, 3, ndf_w1, " \
        "ndf_w3);" in code
    assert "weights" not in code
    assert "np_xy" not in code
    assert "np_z" not in code


def test_lfric_kokkos_trans_accepts_a_kernel_with_both_shapes(
        both_shapes_target):
    """A kernel may ask for quadrature and an evaluator at once.

    Each function space then carries two basis arrays of different rank, and
    the quadrature rule is still appended once, after both of them.
    """
    _, loop, kernel = both_shapes_target
    assert kernel.eval_shapes == ["gh_quadrature_xyoz", "gh_evaluator"]
    code = LFRicKokkosTrans().apply(loop)

    assert "basis_w3_qr(basis_w3_qr_data, 1, ndf_w3, np_xy, np_z);" in code
    assert "basis_w3_on_w3(basis_w3_on_w3_data, 1, ndf_w3, ndf_w3);" in code
    assert "basis_w1_qr(basis_w1_qr_data, 3, ndf_w1, np_xy, np_z);" in code
    assert "basis_w1_on_w3(basis_w1_on_w3_data, 3, ndf_w1, ndf_w3);" in code
    assert code.count("const int np_xy") == 1


def test_lfric_kokkos_trans_accepts_a_diff_basis(diff_basis_target):
    """A derivative has a leading extent of its own, not the basis's.

    On W3 a basis has one component and its derivative three. Reading the
    leading extent from the basis and reusing it would give the derivative
    View a third of its real length, which reads inside the allocation and
    returns the wrong numbers.
    """
    _, loop, _ = diff_basis_target
    code = LFRicKokkosTrans().apply(loop)

    assert "basis_w3(basis_w3_data, 1, ndf_w3, np_xy, np_z);" in code
    assert "diff_basis_w3(diff_basis_w3_data, 3, ndf_w3, np_xy, np_z);" in code


def test_lfric_kokkos_trans_basis_indexing_matches_the_kernel(
        quadrature_target):
    """The generated subscripts keep the Fortran's order.

    A basis array is the one place where a transposed index is invisible: the
    array is square in neither dimension but every subscript is in range, so
    the region compiles, runs and returns the wrong answer. The kernel writes
    ``basis_w3(1,df,qp1,qp2)``, so the region must too, each subscript
    rebased from 1 to 0 and nothing reordered.
    """
    _, loop, _ = quadrature_target
    code = LFRicKokkosTrans().apply(loop)

    assert "basis_w3((1 - 1), (df - 1), (qp1 - 1), (qp2 - 1))" in code
    assert "weights_xy((qp1 - 1))" in code
    assert "weights_z((qp2 - 1))" in code


_FACE_AND_XYOZ_ALGORITHM = """
program kokkos_face_and_xyoz_test
  use field_mod, only : field_type
  use quadrature_xyoz_mod, only : quadrature_xyoz_type
  use quadrature_face_mod, only : quadrature_face_type
  use two_rules_kernel_mod, only : two_rules_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  type(quadrature_xyoz_type) :: qr
  type(quadrature_face_type) :: qrf
  call invoke(two_rules_kernel_type(out_field, in_field, qr, qrf))
end program kokkos_face_and_xyoz_test
"""


# A kernel naming both quadrature shapes. Each function space asking for a
# basis then carries two arrays of the same rank but different extents, and
# each rule appends its own counts and weights, in the order 'gh_shape' names
# the shapes. The two are told apart by their extents alone, which is why
# this is the case that would catch a rule whose arguments were built once
# and reused.
_FACE_AND_XYOZ_KERNEL = """
module two_rules_kernel_mod
  use argument_mod, only : arg_type, func_type, gh_field, gh_real, gh_write, &
                           gh_read, gh_basis, cell_column,                   &
                           gh_quadrature_XYoZ, gh_quadrature_face
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: two_rules_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_write, w3),                        &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    type(func_type) :: meta_funcs(1) = (/                                  &
         func_type(w3, gh_basis) /)
    integer :: gh_shape(2) = (/ gh_quadrature_XYoZ, gh_quadrature_face /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: two_rules_code
  end type two_rules_kernel_type
contains
  subroutine two_rules_code(nlayers, field_out, field_in,                   &
                            ndf_w3, undf_w3, map_w3, basis_w3_qr,           &
                            basis_w3_faces, np_xy, np_z, weights_xy,        &
                            weights_z, nfaces, nqp_faces, wqp_faces)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: np_xy, np_z, nfaces, nqp_faces
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    real(kind=r_def), dimension(np_xy), intent(in) :: weights_xy
    real(kind=r_def), dimension(np_z), intent(in) :: weights_z
    real(kind=r_def), dimension(nqp_faces,nfaces), intent(in) :: wqp_faces
    real(kind=r_def), dimension(1,ndf_w3,np_xy,np_z), intent(in) ::         &
                                                              basis_w3_qr
    real(kind=r_def), dimension(1,ndf_w3,nqp_faces,nfaces), intent(in) ::   &
                                                           basis_w3_faces
    integer(kind=i_def) :: k, df, qp1, qp2, qp, face
    real(kind=r_def) :: total
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        total = 0.0_r_def
        do qp2 = 1, np_z
          do qp1 = 1, np_xy
            total = total + weights_xy(qp1) * weights_z(qp2)               &
                  * basis_w3_qr(1,df,qp1,qp2)
          end do
        end do
        do face = 1, nfaces
          do qp = 1, nqp_faces
            total = total + wqp_faces(qp,face)                             &
                  * basis_w3_faces(1,df,qp,face)
          end do
        end do
        field_out(map_w3(df) + k) = total * field_in(map_w3(df) + k)
      end do
    end do
  end subroutine two_rules_code
end module two_rules_kernel_mod
"""


@pytest.fixture(name="face_quadrature_target")
# pylint: disable-next=unused-argument
def face_quadrature_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads face-quadrature basis data."""
    return _invoke(
        tmp_path, "face_weight", _FACE_QUADRATURE_ALGORITHM,
        _FACE_QUADRATURE_KERNEL)


@pytest.fixture(name="face_and_xyoz_target")
# pylint: disable-next=unused-argument
def face_and_xyoz_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel asks for both quadrature shapes."""
    return _invoke(
        tmp_path, "two_rules", _FACE_AND_XYOZ_ALGORITHM,
        _FACE_AND_XYOZ_KERNEL)


def test_face_quadrature_arguments_are_declared(face_quadrature_target):
    """Face quadrature crosses the ABI as counts, weights and basis data.

    The rule adds three formals rather than XYoZ's four -- a face count and a
    point count by value, and one *rank-2* weight array over the two of them
    -- and each function space asking for a basis adds a rank-4 View shaped
    ``(dim, ndf, np_xyz, nfaces)``. The weights are the one place face
    quadrature is not XYoZ quadrature with different names: XYoZ hands over
    two rank-1 arrays and this hands over one rank-2 array, whose leading
    extent is the point count and so is the stride that has to be exact under
    ``LayoutLeft``.

    None of it is per-cell, so every one of them is an unmanaged View over
    the whole array the PSy layer computed before the loop, and the
    ``bind(C)`` interface declares each as an assumed-size dummy of the
    matching ``iso_c_binding`` kind.
    """
    psy, loop, kernel = face_quadrature_target
    assert kernel.eval_shapes == ["gh_quadrature_face"]
    code = LFRicKokkosTrans().apply(loop)

    assert "const int nfaces" in code
    assert "const int np_xyz" in code
    assert "Kokkos::View<const double**, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> weights_xyz(weights_xyz_data, np_xyz, nfaces);" in code
    assert "Kokkos::View<const double****, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> basis_w3(basis_w3_data, 1, ndf_w3, np_xyz, nfaces);" in code
    assert "weights_xyz((qp - 1), (face - 1))" in code
    assert "basis_w3((1 - 1), (df - 1), (qp - 1), (face - 1))" in code

    # The rule goes last, after the per-function-space arguments, and the
    # face count precedes the point count as 'nfaces_qr, np_xyz_qr' does in
    # the Fortran call the PSy layer would otherwise have written.
    signature = code.split(") {\n")[0]
    parameters = [
        parameter.strip().split()[-1].lstrip("*")
        for parameter in signature.split(",") if parameter.strip().split()]
    assert parameters.index("basis_w3_data") < parameters.index("nfaces")
    assert parameters.index("nfaces") < parameters.index("np_xyz")
    assert parameters.index("np_xyz") < parameters.index("weights_xyz_data")

    fortran = str(psy.gen)
    assert "integer(c_int), value :: nfaces" in fortran
    assert "integer(c_int), value :: np_xyz" in fortran
    assert "real(c_double), dimension(*), intent(in) :: weights_xyz" in fortran
    assert "real(c_double), dimension(*), intent(in) :: basis_w3" in fortran


def test_face_quadrature_call_passes_the_psy_arrays(face_quadrature_target):
    """The PSy layer passes the arrays the quadrature proxy holds.

    Face quadrature's weights are a rank-2 pointer component,
    ``qr_proxy%weights_xyz``, and the basis array is filled by
    ``compute_function`` before the loop. Both are whole-array data, so
    passing either sliced would give the region a cell dimension it does not
    have.
    """
    psy, loop, _ = face_quadrature_target
    LFRicKokkosTrans().apply(loop)
    generated = str(psy.gen).lower()

    call = [line for line in generated.splitlines()
            if "call face_weight_kokkos(" in line]
    assert call, generated
    arguments = call[0].split("(", 1)[1]
    for actual in ("basis_w3_qr", "nfaces_qr", "np_xyz_qr", "weights_xyz_qr"):
        assert actual in arguments
    assert "basis_w3_qr(" not in arguments
    assert "nfaces_qr = qr_proxy%nfaces" in generated
    assert "weights_xyz_qr => qr_proxy%weights_xyz" in generated


def test_face_and_xyoz_together(face_and_xyoz_target):
    """A kernel may ask for face and XYoZ quadrature at once.

    Its one function space then carries two basis arrays of the same rank,
    told apart by their extents alone, and each rule appends its own counts
    and weights in the order ``gh_shape`` names the shapes. A region that
    described the second rule from the first would generate two Views of
    identical extents here, and every subscript of the wrong one would still
    be in range.
    """
    _, loop, kernel = face_and_xyoz_target
    assert kernel.eval_shapes == ["gh_quadrature_xyoz", "gh_quadrature_face"]
    code = LFRicKokkosTrans().apply(loop)

    assert "basis_w3_qr(basis_w3_qr_data, 1, ndf_w3, np_xy, np_z);" in code
    assert "basis_w3_faces(basis_w3_faces_data, 1, ndf_w3, nqp_faces, " \
        "nfaces);" in code
    assert "weights_xy(weights_xy_data, np_xy);" in code
    assert "weights_z(weights_z_data, np_z);" in code
    assert "wqp_faces(wqp_faces_data, nqp_faces, nfaces);" in code

    signature = code.split(") {\n")[0]
    parameters = [
        parameter.strip().split()[-1].lstrip("*")
        for parameter in signature.split(",") if parameter.strip().split()]
    assert parameters.index("basis_w3_qr_data") < parameters.index(
        "basis_w3_faces_data")
    assert parameters.index("weights_z_data") < parameters.index("nfaces")
