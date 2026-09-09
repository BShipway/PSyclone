# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Fixtures shared by more than one of the LFRicKokkosTrans test modules."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _ALGORITHM, _ALLOCATABLE_MODULE_VARIABLE_KERNEL, _KERNEL, _LEVEL_ALGORITHM,
    _LEVEL_KERNEL, _LITERAL_LOCAL_KERNEL, _LOCAL_ALGORITHM, _LOCAL_KERNEL,
    _OPERATOR_ALGORITHM, _OPERATOR_KERNEL, _SECOND_KERNEL, _SECTION_ALGORITHM,
    _SECTION_KERNEL, _SHARED_WRITE_ALGORITHM, _SHARED_WRITE_KERNEL,
    _SOLVER_ALGORITHM, _SOLVER_KERNEL, _TARGET_CALLEE_KERNEL,
    _TARGET_DUMMY_KERNEL, _TARGET_HELPER_MODULE, _TARGET_TWIN_ALGORITHM,
    _TARGET_TWIN_KERNEL, _POINTER_DUMMY_KERNEL,
    _UNRENDERABLE_ORIGIN_KERNEL, _ZERO_BASED_LOCAL_KERNEL, _invoke)


# An invoke whose loop runs into the halo without being asked to. LFRic
# assembles an operator redundantly to the first halo depth, so a kernel
# writing one is bounded by 'mesh%get_last_halo_cell(1)' rather than by the
# owned cells -- which is what two hundred and twenty-one of the loops the
# coverage survey records under 'halo-depth' look like, and what makes this
# fixture the released population rather than a construction.
_HALO_OPERATOR_ALGORITHM = """
program kokkos_halo_operator_test
  use constants_mod,  only : r_def
  use field_mod,      only : field_type
  use operator_mod,   only : operator_type
  use operator_setval_x_kernel_mod, only : operator_setval_x_kernel_type
  implicit none
  type(operator_type) :: op
  type(field_type) :: weight
  call invoke(operator_setval_x_kernel_type(op, weight, 1.0_r_def))
end program kokkos_halo_operator_test
"""


# operator_setval_x's shape with a weight field added. The field is what puts
# a dofmap in the region, and the dofmap is the argument whose View is sized
# by the cell count: a launch reaching into the halo reads columns of it that
# a View sliced by the owned count would not hold.
_HALO_OPERATOR_KERNEL = """
module operator_setval_x_kernel_mod
  use argument_mod, only : arg_type, gh_operator, gh_field, gh_scalar,     &
                           gh_real, gh_write, gh_read, cell_column,        &
                           any_discontinuous_space_1,                      &
                           any_discontinuous_space_2
  use constants_mod, only : i_def, r_def
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: operator_setval_x_kernel_type
    type(arg_type) :: meta_args(3) = (/                                    &
         arg_type(gh_operator, gh_real, gh_write,                          &
                  any_discontinuous_space_1,                               &
                  any_discontinuous_space_2),                              &
         arg_type(gh_field,    gh_real, gh_read,                           &
                  any_discontinuous_space_1),                              &
         arg_type(gh_scalar,   gh_real, gh_read) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: operator_setval_x_code
  end type operator_setval_x_kernel_type
contains
  subroutine operator_setval_x_code(cell, nlayers, ncell_3d, op, weight, &
                                    scalar, ndf1, undf1, map1, ndf2)
    integer(kind=i_def), intent(in) :: cell, nlayers, ncell_3d
    integer(kind=i_def), intent(in) :: ndf1, undf1, ndf2
    integer(kind=i_def), dimension(ndf1), intent(in) :: map1
    real(kind=r_def), dimension(ncell_3d,ndf1,ndf2), intent(inout) :: op
    real(kind=r_def), dimension(undf1), intent(in) :: weight
    real(kind=r_def), intent(in) :: scalar
    integer(kind=i_def) :: df1, df2, k, ik
    do k = 0, nlayers - 1
      ik = (cell - 1) * nlayers + k + 1
      do df2 = 1, ndf2
        do df1 = 1, ndf1
          op(ik, df1, df2) = scalar * weight(map1(df1) + k)
        end do
      end do
    end do
  end subroutine operator_setval_x_code
end module operator_setval_x_kernel_mod
"""


_SECOND_ALGORITHM = """
program kokkos_second_test
  use constants_mod, only : r_tran
  use field_mod, only : field_type
  use scaled_copy_kernel_mod, only : scaled_copy_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  real(kind=r_tran) :: scaling
  call invoke(scaled_copy_kernel_type(out_field, in_field, scaling))
end program kokkos_second_test
"""


# The continuous-write shape, in the smallest form that carries both halves
# of it. 'flux' is 'gh_write' onto W2, whose dofs the neighbouring cells
# share, so two cells of one launch store to the same element; 'out' is
# 'gh_write' onto W3, which no other cell touches. LFRic permits the first
# because the kernel author guarantees the two cells store the same value,
# and neither the metadata nor the body states that guarantee -- so the
# transformation reads the sharing and not the guarantee. It is the shape
# 'ffsl_unify_flux_kernel_code' has in GungHo, which is where the survey
# found it.
_CONTINUOUS_WRITE_KERNEL = """
module flux_probe_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w2, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: flux_probe_kernel_type
    type(arg_type) :: meta_args(3) = (/                        &
         arg_type(gh_field, gh_real, gh_write, w2),            &
         arg_type(gh_field, gh_real, gh_write, w3),            &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: flux_probe_code
  end type flux_probe_kernel_type
contains
  subroutine flux_probe_code(nlayers, flux, out, src, &
                             ndf_w2, undf_w2, map_w2, &
                             ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w2, undf_w2
    integer(kind=i_def), intent(in) :: ndf_w3, undf_w3
    real(kind=r_def), dimension(undf_w2), intent(inout) :: flux
    real(kind=r_def), dimension(undf_w3), intent(inout) :: out
    real(kind=r_def), dimension(undf_w3), intent(in) :: src
    integer(kind=i_def), dimension(ndf_w2), intent(in) :: map_w2
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w2
        flux(map_w2(df) + k) = 2.0_r_def*src(map_w3(1) + k)
      end do
      do df = 1, ndf_w3
        out(map_w3(df) + k) = src(map_w3(df) + k)
      end do
    end do
  end subroutine flux_probe_code
end module flux_probe_kernel_mod
"""


_CONTINUOUS_WRITE_ALGORITHM = """
program kokkos_continuous_write_test
  use field_mod, only : field_type
  use flux_probe_kernel_mod, only : flux_probe_kernel_type
  implicit none
  type(field_type) :: flux, src, out
  call invoke(flux_probe_kernel_type(flux, out, src))
end program kokkos_continuous_write_test
"""


# The one store to a shared dof that no atomic carries: the value reads back
# the element it is replacing. 'Kokkos::atomic_store' makes the write
# indivisible and says nothing at all about the read that preceded it, so a
# cell can store a value computed from what the neighbouring cell has since
# overwritten. The scaling is written with the target under a product rather
# than at the top of one, because 'flux = 2.0*flux' is a read-modify-write an
# atomic does carry and would be answered rather than refused.
_CONTINUOUS_READ_BACK_KERNEL = _CONTINUOUS_WRITE_KERNEL.replace(
    "flux(map_w2(df) + k) = 2.0_r_def*src(map_w3(1) + k)",
    "flux(map_w2(df) + k) = 2.0_r_def*flux(map_w2(df) + k)"
    "*src(map_w3(1) + k)")


# The shared write in the shape GungHo writes it most often: 'matrix_vector',
# an operator applied to a field and accumulated onto a continuous space. The
# operator is what makes this different from 'inc_probe_kernel_mod' -- LFRic
# passes the cell index as the first actual for exactly the kernels that take
# one, and a colouring rewrites that actual into a lookup in the colour map.
_SHARED_WRITE_OPERATOR_KERNEL = """
module inc_operator_probe_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_operator, gh_real,      &
                           gh_inc, gh_read, cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w2, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: inc_operator_probe_kernel_type
    type(arg_type) :: meta_args(3) = (/                        &
         arg_type(gh_field,    gh_real, gh_inc,  w2),          &
         arg_type(gh_field,    gh_real, gh_read, w3),          &
         arg_type(gh_operator, gh_real, gh_read, w2, w3) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: inc_operator_probe_code
  end type inc_operator_probe_kernel_type
contains
  subroutine inc_operator_probe_code(cell, nlayers, lhs, x, ncell_3d, &
                                     matrix, ndf_w2, undf_w2, map_w2, &
                                     ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: cell, nlayers, ncell_3d
    integer(kind=i_def), intent(in) :: ndf_w2, undf_w2, ndf_w3, undf_w3
    real(kind=r_def), dimension(undf_w2), intent(inout) :: lhs
    real(kind=r_def), dimension(undf_w3), intent(in) :: x
    real(kind=r_def), dimension(ncell_3d,ndf_w2,ndf_w3), intent(in) :: matrix
    integer(kind=i_def), dimension(ndf_w2), intent(in) :: map_w2
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: df1, df2, k, ik
    do k = 0, nlayers - 1
      ik = (cell - 1) * nlayers + k + 1
      do df1 = 1, ndf_w2
        do df2 = 1, ndf_w3
          lhs(map_w2(df1) + k) = lhs(map_w2(df1) + k) + &
                                 matrix(ik, df1, df2) * x(map_w3(df2) + k)
        end do
      end do
    end do
  end subroutine inc_operator_probe_code
end module inc_operator_probe_kernel_mod
"""


_SHARED_WRITE_OPERATOR_ALGORITHM = """
program kokkos_shared_write_operator_test
  use field_mod, only : field_type
  use operator_mod, only : operator_type
  use inc_operator_probe_kernel_mod, only : inc_operator_probe_kernel_type
  implicit none
  type(field_type) :: lhs, x
  type(operator_type) :: matrix
  call invoke(inc_operator_probe_kernel_type(lhs, x, matrix))
end program kokkos_shared_write_operator_test
"""


# The same kernel with one array sized by a module constant instead of by a
# formal. Fortran allows it -- a module entity is as valid an automatic bound
# as a dummy -- but the launch computes its scratch size before it enters the
# region, so an extent it cannot name there cannot be sized.
_UNSIZED_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : n_moist").replace(
    "dimension(nlayers) :: swept", "dimension(n_moist) :: swept")


# The same kernel with arithmetic in a bound. `dimension(nlayers+1)` is the
# second tier of what a kernel author writes: still sized from a formal, but
# not by naming one.
_ARITHMETIC_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(nlayers+1) :: swept")


# The same kernel with a lower bound written out and equal to 1. The rule is
# about the rendered value, not the source text, so this is tier 1 wearing the
# syntax of a tier the transformation refuses.
_EXPLICIT_ONE_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(1:nlayers) :: swept")


# The same kernel with a lower bound that is not 1, and an upper bound that
# is an expression. Both the extent and the origin are read from the one
# declaration, so a shape that moves the origin and computes the extent at
# once is the case where reading them apart would disagree.
_LOWER_BOUND_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(0:nlayers-1) :: swept")


# An array centred on zero rather than based at it. This is the shape stage
# 3b's defect note names: the extent and the origin are different expressions
# over the same name, so a region that sized the View correctly and shifted
# its subscripts by the Fortran default would index a View of the right size
# from the wrong place -- a wrong answer rather than a refusal.
_NEGATIVE_ORIGIN_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept",
    "dimension(-nlayers:nlayers) :: u_e").replace("swept(", "u_e(")


# A second local sized by a `parameter` the kernel declares beside it. This is
# the commonest shape the catalogue's `local-array` row counts:
# `integer(kind=i_def), parameter :: nfaces = 4` sizing
# `real(kind=r_tran), dimension(nfaces) :: v_dot_n` in five GungHo kernels.
# The value is in the Fortran, so the extent is resolved to it rather than
# refused for naming something the launch has no argument for.
_NAMED_CONSTANT_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k",
    "    integer(kind=i_def), parameter :: nfaces = 4\n"
    "    integer(kind=i_def) :: k").replace(
    "    real(kind=r_def), dimension(nlayers) :: swept",
    "    real(kind=r_def), dimension(nfaces) :: v_dot_n\n"
    "    real(kind=r_def), dimension(nlayers) :: swept").replace(
    "    swept(nlayers) = partial(nlayers)",
    "    do k = 1, nfaces\n"
    "      v_dot_n(k) = partial(1)\n"
    "    end do\n"
    "    swept(nlayers) = partial(nlayers) + v_dot_n(nfaces)")


# A second local whose extent divides. `dimension((stencil_size + 1) / 2)` is
# what hori_dep_dist_midpoint_kernel_mod declares and what the catalogue's
# `local-array` row counts twice. Fortran and C++ both truncate an integer
# quotient toward zero, so the extent is carried rather than refused, and the
# launch says so where it sizes the scratch.
_DIVIDED_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "    real(kind=r_def), dimension(nlayers) :: swept",
    "    real(kind=r_def), dimension((nlayers + 1)/2) :: u_local\n"
    "    real(kind=r_def), dimension(nlayers) :: swept").replace(
    "    swept(nlayers) = partial(nlayers)",
    "    do k = 1, (nlayers + 1)/2\n"
    "      u_local(k) = partial(k)\n"
    "    end do\n"
    "    swept(nlayers) = partial(nlayers) + u_local(1)")


# A declared shape the C writer has no way to render at all. The back-end's
# own failure is a VisitorError, which `validate` may not raise, so it is
# turned into a refusal that names the array.
#
# MODULO is the intrinsic used, because it has no C spelling: C's `%` is MOD,
# which differs from MODULO for a negative operand, and the writer has no
# entry for it. It used to be MAX -- ffsl_flux_z_nirvana_kernel_mod declares
# `field_local_upper(MAX(nlayers-monotone_above,1), 3)`, which is where the
# refusal was found -- but an integer MAX is now written as `std::max` and is
# no longer unwritable. The property under test is unchanged: a shape the
# writer refuses is reported by `validate`, not raised from inside `apply`.
_UNWRITABLE_SHAPE_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(modulo(nlayers,3)) :: swept")


# The shape that used to be unwritable. ffsl_flux_z_nirvana_kernel_mod
# declares `field_local_upper(MAX(nlayers-monotone_above,1), 3)`, and an
# integer MAX is now written as `std::max`, so this is carried rather than
# refused: the scratch is sized by the call and the launch is told to
# evaluate `nlayers` alone.
_MAXIMUM_SHAPE_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(max(nlayers-1,1)) :: swept")


# Arithmetic over a module constant rather than over a formal. Accepting
# expressions must not have widened *which names* may appear in one: the
# scratch size is still computed where only kernel arguments are in scope.
_UNSIZED_EXPRESSION_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : n_moist").replace(
    "dimension(nlayers) :: swept", "dimension(n_moist+1) :: swept")


# The same kernel with a third local of a kind the ABI does not name. The
# refusal is the one _validate_formals already makes for a formal, asked of a
# local: the scratch View has to have a C element type.
_UNMAPPED_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, l_def, r_def").replace(
    "    real(kind=r_def), dimension(nlayers) :: swept",
    "    real(kind=r_def), dimension(nlayers) :: swept\n"
    "    logical(kind=l_def), dimension(nlayers) :: rising").replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      rising(k) = swept(k) > partial(k)\n"
    "      if (rising(k)) then\n"
    "        field_out(map_w3(1) + k - 1) = swept(k)\n"
    "      end if")


# The same kernel asking three shape enquiries of arrays it already declares.
# LBOUND and UBOUND of a local, and SIZE of a formal, all of which the
# declaration answers: the region never asks a View for a shape the Fortran
# has stated.
_BOUND_KERNEL = _LOCAL_KERNEL.replace(
    "    do k = 2, nlayers",
    "    do k = lbound(partial, 1) + 1, ubound(partial, 1)").replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = swept(k) + size(field_in)")


# SIZE of the same rank-2 local without naming a dimension. Valid Fortran --
# it returns the total element count -- but not a bound of any one dimension,
# and the region has no way to say it.
_WHOLE_SIZE_KERNEL = _LITERAL_LOCAL_KERNEL.replace(
    "      swept(k,1) = swept(k + 1,1) - partial(k)",
    "      swept(k,1) = swept(k + 1,1) - partial(k) * size(swept)")


# SIZE of a scalar formal. fparser2 parses it, since the check that would
# refuse it is semantic rather than syntactic, so the transformation is what
# has to notice.
_SCALAR_BOUND_KERNEL = _LOCAL_KERNEL.replace(
    "    do k = 2, nlayers", "    do k = 2, size(nlayers)")


# UBOUND of one element of an array rather than of the array. An
# ArrayReference is a Reference, so the test has to be for the exact type or
# an element's bound is read as the array's.
_ELEMENT_BOUND_KERNEL = _LOCAL_KERNEL.replace(
    "    do k = 2, nlayers", "    do k = 2, ubound(map_w3(1), 1)")


# A dimension given by a variable. The substitution is symbolic, against the
# declaration, so it has to know which entry of the shape to take before the
# region runs.
_VARIABLE_BOUND_KERNEL = _LOCAL_KERNEL.replace(
    "    do k = 2, nlayers", "    do k = 2, ubound(partial, k)")


# A dimension outside the declared rank. Fortran would refuse it, but nothing
# between the source and the backend does.
_OUT_OF_RANGE_BOUND_KERNEL = _LOCAL_KERNEL.replace(
    "    do k = 2, nlayers", "    do k = 2, ubound(partial, 2)")


@pytest.fixture(name="target")
# pylint: disable-next=unused-argument
def target_fixture(tmp_path, clear_module_manager_instance):
    """Create the production metadata/body in a minimal LFRic invoke."""
    return _invoke(tmp_path, "moist_dyn_gas", _ALGORITHM, _KERNEL)


@pytest.fixture(name="second_target")
# pylint: disable-next=unused-argument
def second_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an unrelated supported kernel in a minimal LFRic invoke."""
    return _invoke(
        tmp_path, "scaled_copy", _SECOND_ALGORITHM, _SECOND_KERNEL)


@pytest.fixture(name="halo_operator_target")
# pylint: disable-next=unused-argument
def halo_operator_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose loop runs to the first halo depth."""
    return _invoke(
        tmp_path, "operator_setval_x", _HALO_OPERATOR_ALGORITHM,
        _HALO_OPERATOR_KERNEL)


@pytest.fixture(name="shared_write_target")
# pylint: disable-next=unused-argument
def shared_write_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel accumulates into a shared dof."""
    return _invoke(
        tmp_path, "inc_probe", _SHARED_WRITE_ALGORITHM, _SHARED_WRITE_KERNEL)


@pytest.fixture(name="continuous_write_target")
# pylint: disable-next=unused-argument
def continuous_write_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel stores to a dof two cells share."""
    return _invoke(
        tmp_path, "flux_probe", _CONTINUOUS_WRITE_ALGORITHM,
        _CONTINUOUS_WRITE_KERNEL)


@pytest.fixture(name="continuous_read_back_target")
# pylint: disable-next=unused-argument
def continuous_read_back_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose store to a shared dof reads that dof."""
    return _invoke(
        tmp_path, "flux_probe", _CONTINUOUS_WRITE_ALGORITHM,
        _CONTINUOUS_READ_BACK_KERNEL)


@pytest.fixture(name="shared_write_operator_target")
# pylint: disable-next=unused-argument
def shared_write_operator_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke accumulating into a shared dof through an operator."""
    return _invoke(
        tmp_path, "inc_operator_probe", _SHARED_WRITE_OPERATOR_ALGORITHM,
        _SHARED_WRITE_OPERATOR_KERNEL)


@pytest.fixture(name="section_target")
# pylint: disable-next=unused-argument
def section_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel assigns a whole-column section."""
    return _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM, _SECTION_KERNEL)


@pytest.fixture(name="solver_target")
# pylint: disable-next=unused-argument
def solver_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel computes in single precision."""
    return _invoke(
        tmp_path, "scaled_solver", _SOLVER_ALGORITHM, _SOLVER_KERNEL)


@pytest.fixture(name="operator_target")
# pylint: disable-next=unused-argument
def operator_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads an LMA operator."""
    return _invoke(
        tmp_path, "dg_matrix_vector", _OPERATOR_ALGORITHM, _OPERATOR_KERNEL)


@pytest.fixture(name="local_target")
# pylint: disable-next=unused-argument
def local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel holds two automatic column arrays."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _LOCAL_KERNEL)


@pytest.fixture(name="level_target")
# pylint: disable-next=unused-argument
def level_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel holds one parallelisable level loop."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _LEVEL_KERNEL)


@pytest.fixture(name="unsized_local_target")
# pylint: disable-next=unused-argument
def unsized_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel sizes a local by a module constant."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _UNSIZED_LOCAL_KERNEL)


@pytest.fixture(name="literal_local_target")
# pylint: disable-next=unused-argument
def literal_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel sizes a local partly by a literal."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _LITERAL_LOCAL_KERNEL)


@pytest.fixture(name="arithmetic_local_target")
# pylint: disable-next=unused-argument
def arithmetic_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel sizes a local by nlayers + 1."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ARITHMETIC_LOCAL_KERNEL)


@pytest.fixture(name="explicit_one_local_target")
# pylint: disable-next=unused-argument
def explicit_one_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel writes out a lower bound of 1."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _EXPLICIT_ONE_LOCAL_KERNEL)


@pytest.fixture(name="lower_bound_local_target")
# pylint: disable-next=unused-argument
def lower_bound_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local from zero."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _LOWER_BOUND_LOCAL_KERNEL)


@pytest.fixture(name="zero_based_local_target")
# pylint: disable-next=unused-argument
def zero_based_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local from zero."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ZERO_BASED_LOCAL_KERNEL)


@pytest.fixture(name="negative_origin_local_target")
# pylint: disable-next=unused-argument
def negative_origin_local_target_fixture(tmp_path,
                                         clear_module_manager_instance):
    """Create an invoke whose kernel centres a local on zero."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _NEGATIVE_ORIGIN_LOCAL_KERNEL)


@pytest.fixture(name="unrenderable_origin_target")
# pylint: disable-next=unused-argument
def unrenderable_origin_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create an invoke whose kernel squares in a declared lower bound."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _UNRENDERABLE_ORIGIN_KERNEL)


@pytest.fixture(name="named_constant_local_target")
# pylint: disable-next=unused-argument
def named_constant_local_target_fixture(tmp_path,
                                        clear_module_manager_instance):
    """Create an invoke whose kernel sizes a local by its own parameter."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _NAMED_CONSTANT_LOCAL_KERNEL)


@pytest.fixture(name="divided_local_target")
# pylint: disable-next=unused-argument
def divided_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel sizes a local by a division."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _DIVIDED_LOCAL_KERNEL)


@pytest.fixture(name="unwritable_shape_target")
# pylint: disable-next=unused-argument
def unwritable_shape_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel takes a MAX in a declared bound."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _UNWRITABLE_SHAPE_KERNEL)


@pytest.fixture(name="maximum_shape_target")
# pylint: disable-next=unused-argument
def maximum_shape_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel sizes a local by an integer MAX."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _MAXIMUM_SHAPE_KERNEL)


@pytest.fixture(name="unsized_expression_target")
# pylint: disable-next=unused-argument
def unsized_expression_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel sizes a local by n_moist + 1."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _UNSIZED_EXPRESSION_KERNEL)


@pytest.fixture(name="unmapped_local_target")
# pylint: disable-next=unused-argument
def unmapped_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel holds a local array of a logical kind."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _UNMAPPED_LOCAL_KERNEL)


@pytest.fixture(name="bound_target")
# pylint: disable-next=unused-argument
def bound_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel asks for LBOUND, UBOUND and SIZE."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _BOUND_KERNEL)


@pytest.fixture(name="whole_size_target")
# pylint: disable-next=unused-argument
def whole_size_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke asking the SIZE of a rank-2 local with no dimension."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _WHOLE_SIZE_KERNEL)


@pytest.fixture(name="scalar_bound_target")
# pylint: disable-next=unused-argument
def scalar_bound_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke asking the SIZE of a scalar formal."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _SCALAR_BOUND_KERNEL)


@pytest.fixture(name="element_bound_target")
# pylint: disable-next=unused-argument
def element_bound_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke asking the UBOUND of one array element."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ELEMENT_BOUND_KERNEL)


@pytest.fixture(name="variable_bound_target")
# pylint: disable-next=unused-argument
def variable_bound_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose UBOUND names its dimension with a variable."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _VARIABLE_BOUND_KERNEL)


@pytest.fixture(name="out_of_range_bound_target")
# pylint: disable-next=unused-argument
def out_of_range_bound_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke asking for a dimension past the declared rank."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _OUT_OF_RANGE_BOUND_KERNEL)


# ---------------------------------------------------------------------------
# Intrinsics: the allocation tier and the validate/apply gap
# ---------------------------------------------------------------------------
# The column solver with its two locals given a deferred shape and an
# ALLOCATE, which is how vertical_cubic_sl_kernel_mod and the poly2d family
# state a size the declaration could have carried. Both extents are kernel
# arguments, so this is the scratch case wearing an ALLOCATE.
_ALLOCATE_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "    real(kind=r_def), dimension(nlayers) :: partial\n"
    "    real(kind=r_def), dimension(nlayers) :: swept\n",
    "    real(kind=r_def), allocatable, dimension(:) :: partial\n"
    "    real(kind=r_def), allocatable, dimension(:) :: swept\n").replace(
    "    partial(1) = field_in(map_w3(1))",
    "    allocate( partial(nlayers), swept(nlayers) )\n"
    "    partial(1) = field_in(map_w3(1))").replace(
    "  end subroutine column_solve_code",
    "    deallocate( partial, swept )\n"
    "  end subroutine column_solve_code")


# The same kernel sizing an allocation from the data rather than from a
# region-entry value, as apply_variable_hx_kernel_mod does with
# `allocate(t(minval(map_wt) : maxval(map_wt) + nlayers - 1))`. The launch
# computes its scratch size on the host, where map_w3 is not a value it can
# reduce over.
_UNKNOWN_ALLOCATE_KERNEL = _ALLOCATE_LOCAL_KERNEL.replace(
    "    allocate( partial(nlayers), swept(nlayers) )",
    "    allocate( partial(minval(map_w3)), swept(nlayers) )")


# An allocation inside a loop, which is not one array with a size but a
# different array each trip.
_LOOPED_ALLOCATE_KERNEL = _ALLOCATE_LOCAL_KERNEL.replace(
    "    allocate( partial(nlayers), swept(nlayers) )",
    "    allocate( swept(nlayers) )\n"
    "    do k = 1, nlayers\n"
    "      allocate( partial(nlayers) )\n"
    "    end do")


# conservative_neg_fix_code's shape: a whole-array MINVAL over a
# kernel-local. This is the loop wave A found accepted by validate and
# refused by the writer, and it is the witness that the gap is closed in the
# accepting direction as well as the refusing one.
_MINVAL_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    real(kind=r_def) :: floor_value\n").replace(
    "    swept(nlayers) = partial(nlayers)",
    "    floor_value = minval(partial)\n"
    "    swept(nlayers) = partial(nlayers) + floor_value")


# The same kernel reading an intrinsic no writer in the chain has an entry
# for. TINY is chosen because nothing else about the kernel changes: the
# refusal has to come from the intrinsic and from nothing else.
_TINY_KERNEL = _MINVAL_KERNEL.replace(
    "    floor_value = minval(partial)",
    "    floor_value = tiny(partial(1))")


@pytest.fixture(name="allocate_local_target")
# pylint: disable-next=unused-argument
def allocate_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel ALLOCATEs its locals at entry."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ALLOCATE_LOCAL_KERNEL)


@pytest.fixture(name="unknown_allocate_target")
# pylint: disable-next=unused-argument
def unknown_allocate_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel sizes an allocation from its data."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _UNKNOWN_ALLOCATE_KERNEL)


@pytest.fixture(name="looped_allocate_target")
# pylint: disable-next=unused-argument
def looped_allocate_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel allocates inside a loop."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _LOOPED_ALLOCATE_KERNEL)


@pytest.fixture(name="minval_target")
# pylint: disable-next=unused-argument
def minval_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reduces a local with MINVAL."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _MINVAL_KERNEL)


@pytest.fixture(name="tiny_target")
# pylint: disable-next=unused-argument
def tiny_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads an intrinsic no writer has."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _TINY_KERNEL)


# An allocation carrying an option beside the arrays it names. `mold=` states
# where the shape and type come from, and a literal is used for it so that the
# refusal has to name an argument that is not a reference to anything.
_OPTION_ALLOCATE_KERNEL = _ALLOCATE_LOCAL_KERNEL.replace(
    "    allocate( partial(nlayers), swept(nlayers) )",
    "    allocate( partial(nlayers), swept(nlayers), mold=0.0_r_def )")


# The same local allocated twice, which two shapes over one name. Scratch is
# reserved once with one size, so this is a refusal rather than a choice
# between the two.
_TWICE_ALLOCATE_KERNEL = _ALLOCATE_LOCAL_KERNEL.replace(
    "    allocate( partial(nlayers), swept(nlayers) )",
    "    allocate( partial(nlayers), swept(nlayers) )\n"
    "    allocate( partial(nlayers) )")


# An allocation naming its array and stating no bounds for it. Fortran would
# not accept this of a deferred-shape array, and the frontend does not judge
# it, so the conversion says why it has no size to reserve rather than
# reserving nothing.
_SHAPELESS_ALLOCATE_KERNEL = _ALLOCATE_LOCAL_KERNEL.replace(
    "    allocate( partial(nlayers), swept(nlayers) )",
    "    allocate( partial, swept(nlayers) )")


# A kernel allocating the module's own workspace rather than its own, which
# is how a module-scope allocatable acquires a shape. The storage outlives the
# region and is shared with whatever else the module lets at it, so it is not
# the automatic array the scratch reservation stands in for.
_MODULE_ALLOCATE_KERNEL = _ALLOCATABLE_MODULE_VARIABLE_KERNEL.replace(
    "    partial(1) = field_in(map_w3(1))",
    "    allocate( profile_heights(nlayers) )\n"
    "    partial(1) = field_in(map_w3(1))").replace(
    "  end subroutine column_solve_code",
    "    deallocate( profile_heights )\n"
    "  end subroutine column_solve_code")


@pytest.fixture(name="option_allocate_target")
# pylint: disable-next=unused-argument
def option_allocate_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel allocates with an option argument."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _OPTION_ALLOCATE_KERNEL)


@pytest.fixture(name="twice_allocate_target")
# pylint: disable-next=unused-argument
def twice_allocate_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel allocates one local twice."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _TWICE_ALLOCATE_KERNEL)


@pytest.fixture(name="shapeless_allocate_target")
# pylint: disable-next=unused-argument
def shapeless_allocate_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel allocates without stating a shape."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _SHAPELESS_ALLOCATE_KERNEL)


@pytest.fixture(name="module_allocate_target")
# pylint: disable-next=unused-argument
def module_allocate_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel allocates a module-scope array."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _MODULE_ALLOCATE_KERNEL)


# ---------------------------------------------------------------------------
# Task E7: a TARGET dummy is inlinable.
# ---------------------------------------------------------------------------

@pytest.fixture(name="target_dummy_target")
# pylint: disable-next=unused-argument
def target_dummy_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke calling a helper whose column is a TARGET."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _TARGET_DUMMY_KERNEL)


@pytest.fixture(name="pointer_dummy_target")
# pylint: disable-next=unused-argument
def pointer_dummy_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke calling a helper whose column is a POINTER."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _POINTER_DUMMY_KERNEL)


@pytest.fixture(name="shared_target_helper")
# pylint: disable-next=unused-argument
def shared_target_helper_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose two kernels call one TARGET-dummy helper."""
    return _invoke(
        tmp_path, "column_solve", _TARGET_TWIN_ALGORITHM,
        _TARGET_CALLEE_KERNEL,
        extra={"column_twin_kernel_mod": _TARGET_TWIN_KERNEL,
               "target_helper_mod": _TARGET_HELPER_MODULE})
