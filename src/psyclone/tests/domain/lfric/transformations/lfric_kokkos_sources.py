# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Algorithm and kernel sources shared by the LFRicKokkosTrans tests."""

# pylint: disable=protected-access

from psyclone.configuration import Config
from psyclone.domain.lfric import LFRicLoop
from psyclone.parse import ModuleManager
from psyclone.parse.algorithm import parse
from psyclone.psyGen import PSyFactory


_ALGORITHM = """
program kokkos_test
  use field_mod, only : field_type
  use moist_dyn_gas_kernel_mod, only : moist_dyn_gas_kernel_type
  implicit none
  type(field_type) :: moist_dyn, mr
  call invoke(moist_dyn_gas_kernel_type(moist_dyn, mr))
end program kokkos_test
"""


_KERNEL = """
module moist_dyn_gas_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : wtheta
  use kernel_mod, only : kernel_type
  use planet_config_mod, only : recip_epsilon
  implicit none
  type, public, extends(kernel_type) :: moist_dyn_gas_kernel_type
    type(arg_type) :: meta_args(2) = (/                              &
         arg_type(gh_field, gh_real, gh_write, wtheta),              &
         arg_type(gh_field, gh_real, gh_read,  wtheta) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: moist_dyn_gas_code
  end type moist_dyn_gas_kernel_type
contains
  subroutine moist_dyn_gas_code(nlayers, moist_dyn_gas, mr_v, &
                                ndf_wtheta, undf_wtheta, map_wtheta)
    integer(kind=i_def), intent(in) :: nlayers, ndf_wtheta, undf_wtheta
    real(kind=r_def), dimension(undf_wtheta), intent(inout) :: moist_dyn_gas
    real(kind=r_def), dimension(undf_wtheta), intent(in) :: mr_v
    integer(kind=i_def), dimension(ndf_wtheta), intent(in) :: map_wtheta
    integer(kind=i_def) :: k, df
    real(kind=r_def) :: mr_v_at_dof
    do k = 0, nlayers - 1
      do df = 1, ndf_wtheta
        mr_v_at_dof = mr_v(map_wtheta(df) + k)
        moist_dyn_gas(map_wtheta(df) + k) = &
            1.0_r_def + recip_epsilon * mr_v_at_dof
      end do
    end do
  end subroutine moist_dyn_gas_code
end module moist_dyn_gas_kernel_mod
"""


# A second, deliberately unrelated shape: a different region name, an r_tran
# scalar argument, two function spaces and so two dofmaps, and no module
# constant. Nothing about it is special-cased, so it only generates if the
# transformation really does derive its contract from the kernel.
_SECOND_KERNEL = """
module scaled_copy_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_scalar, gh_real, gh_write, &
                           gh_read, cell_column
  use constants_mod, only : i_def, r_def, r_tran
  use fs_continuity_mod, only : w3, wtheta
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: scaled_copy_kernel_type
    type(arg_type) :: meta_args(3) = (/                              &
         arg_type(gh_field,  gh_real, gh_write, w3),                 &
         arg_type(gh_field,  gh_real, gh_read,  wtheta),             &
         arg_type(gh_scalar, gh_real, gh_read) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: scaled_copy_code
  end type scaled_copy_kernel_type
contains
  subroutine scaled_copy_code(nlayers, field_out, field_in, scaling, &
                              ndf_w3, undf_w3, map_w3, &
                              ndf_wtheta, undf_wtheta, map_wtheta)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: ndf_wtheta, undf_wtheta
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_wtheta), intent(in) :: field_in
    real(kind=r_tran), intent(in) :: scaling
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def), dimension(ndf_wtheta), intent(in) :: map_wtheta
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        field_out(map_w3(df) + k) = &
            scaling * field_in(map_wtheta(df) + k)
      end do
    end do
  end subroutine scaled_copy_code
end module scaled_copy_kernel_mod
"""


# The shared-write shape, in the smallest form that carries both halves of
# it. 'acc' is 'gh_inc' onto W2 and its statement is a read-modify-write, so
# two cells sharing one of its dofs both accumulate into it; 'out' is
# 'gh_write' onto W3, which no other cell touches. A kernel with only the
# first could not tell an atomic applied per argument from one applied to
# every write the region makes.
_SHARED_WRITE_KERNEL = """
module inc_probe_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_inc, gh_read, &
                           gh_write, cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w2, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: inc_probe_kernel_type
    type(arg_type) :: meta_args(3) = (/                        &
         arg_type(gh_field, gh_real, gh_inc,   w2),            &
         arg_type(gh_field, gh_real, gh_write, w3),            &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: inc_probe_code
  end type inc_probe_kernel_type
contains
  subroutine inc_probe_code(nlayers, acc, out, src, &
                            ndf_w2, undf_w2, map_w2, &
                            ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w2, undf_w2
    integer(kind=i_def), intent(in) :: ndf_w3, undf_w3
    real(kind=r_def), dimension(undf_w2), intent(inout) :: acc
    real(kind=r_def), dimension(undf_w3), intent(inout) :: out
    real(kind=r_def), dimension(undf_w3), intent(in) :: src
    integer(kind=i_def), dimension(ndf_w2), intent(in) :: map_w2
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        out(map_w3(df) + k) = 2.0_r_def*src(map_w3(df) + k)
      end do
      do df = 1, ndf_w2
        acc(map_w2(df) + k) = acc(map_w2(df) + k) + src(map_w3(1) + k)
      end do
    end do
  end subroutine inc_probe_code
end module inc_probe_kernel_mod
"""


_SHARED_WRITE_ALGORITHM = """
program kokkos_shared_write_test
  use field_mod, only : field_type
  use inc_probe_kernel_mod, only : inc_probe_kernel_type
  implicit none
  type(field_type) :: acc, src, out
  call invoke(inc_probe_kernel_type(acc, out, src))
end program kokkos_shared_write_test
"""


# A whole-column section in the shape the finite-volume family uses, reduced
# to the part that matters: a slice assigned as a unit, with a different lower
# bound on each side so that a lowering which ignored the offsets would give a
# visibly wrong answer rather than an accidentally right one.
_SECTION_ALGORITHM = """
program kokkos_section_test
  use field_mod, only : field_type
  use fv_difference_kernel_mod, only : fv_difference_kernel_type
  implicit none
  type(field_type) :: difference, mass_flux
  call invoke(fv_difference_kernel_type(difference, mass_flux))
end program kokkos_section_test
"""


_SECTION_KERNEL = """
module fv_difference_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_readwrite, &
                           gh_read, cell_column
  use constants_mod, only : i_def, r_tran
  use fs_continuity_mod, only : w3, w2v
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: fv_difference_kernel_type
    type(arg_type) :: meta_args(2) = (/                              &
         arg_type(gh_field, gh_real, gh_readwrite, w3),              &
         arg_type(gh_field, gh_real, gh_read,      w2v) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: fv_difference_code
  end type fv_difference_kernel_type
contains
  subroutine fv_difference_code(nlayers, difference, mass_flux, &
                                ndf_w3, undf_w3, map_w3,        &
                                ndf_w2v, undf_w2v, map_w2v)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: ndf_w2v, undf_w2v
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def), dimension(ndf_w2v), intent(in) :: map_w2v
    real(kind=r_tran), dimension(undf_w3), intent(inout) :: difference
    real(kind=r_tran), dimension(undf_w2v), intent(in) :: mass_flux
    integer(kind=i_def) :: nl, w3_idx, b_idx
    w3_idx = map_w3(1)
    b_idx = map_w2v(1)
    nl = nlayers - 1
    difference(w3_idx : w3_idx + nl) = &
        mass_flux(b_idx + 1 : b_idx + nl + 1) - mass_flux(b_idx : b_idx + nl)
  end subroutine fv_difference_code
end module fv_difference_kernel_mod
"""


# The same kernel with the written field read back at a shifted offset. In
# Fortran the right-hand side is evaluated before any of it is assigned, so
# the statement is well defined; a loop written naively over the column is
# not. The lowering refuses it, and so the capture must.
_DEPENDENT_SECTION_KERNEL = _SECTION_KERNEL.replace(
    "mass_flux(b_idx + 1 : b_idx + nl + 1) - mass_flux(b_idx : b_idx + nl)",
    "difference(w3_idx + 1 : w3_idx + nl + 1) - mass_flux(b_idx : b_idx + nl)")


# Single precision, in the shape the solver family uses. The fields are
# r_solver and the scalar is r_single -- two different 4-byte kinds, so a
# widening that hardcoded one of them would not generate this. The local and
# the literal are the point: neither crosses the C ABI, so nothing but the
# generated declaration and the generated literal decides what precision the
# region computes in.
_SOLVER_ALGORITHM = """
program kokkos_solver_test
  use constants_mod, only : r_single
  use r_solver_field_mod, only : r_solver_field_type
  use scaled_solver_kernel_mod, only : scaled_solver_kernel_type
  implicit none
  type(r_solver_field_type) :: out_field, in_field
  real(kind=r_single) :: scaling
  call invoke(scaled_solver_kernel_type(out_field, in_field, scaling))
end program kokkos_solver_test
"""


_SOLVER_KERNEL = """
module scaled_solver_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_scalar, gh_real, gh_write, &
                           gh_read, cell_column
  use constants_mod, only : i_def, r_single, r_solver
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: scaled_solver_kernel_type
    type(arg_type) :: meta_args(3) = (/                              &
         arg_type(gh_field,  gh_real, gh_write, w3),                 &
         arg_type(gh_field,  gh_real, gh_read,  w3),                 &
         arg_type(gh_scalar, gh_real, gh_read) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: scaled_solver_code
  end type scaled_solver_kernel_type
contains
  subroutine scaled_solver_code(nlayers, field_out, field_in, scaling, &
                                ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    real(kind=r_solver), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_solver), dimension(undf_w3), intent(in) :: field_in
    real(kind=r_single), intent(in) :: scaling
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df
    real(kind=r_solver) :: scaled
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        scaled = 2.0_r_solver * field_in(map_w3(df) + k)
        field_out(map_w3(df) + k) = scaling * scaled
      end do
    end do
  end subroutine scaled_solver_code
end module scaled_solver_kernel_mod
"""


# A column solve reduced to its shape: two automatic arrays over nlayers, a
# forward sweep and a backward one. It is sci_tri_solve_kernel_mod's structure
# without its algebra. The arrays are what the region has to place in team
# scratch -- their extent is a runtime value, so neither can be a C++ local --
# and the descending loop is what makes a launch that ignored the step sign
# return the forward sweep's intermediates instead of failing.
_LOCAL_ALGORITHM = """
program kokkos_local_test
  use field_mod, only : field_type
  use column_solve_kernel_mod, only : column_solve_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  call invoke(column_solve_kernel_type(out_field, in_field))
end program kokkos_local_test
"""


_LOCAL_KERNEL = """
module column_solve_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: column_solve_kernel_type
    type(arg_type) :: meta_args(2) = (/                              &
         arg_type(gh_field, gh_real, gh_write, w3),                  &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: column_solve_code
  end type column_solve_kernel_type
contains
  subroutine column_solve_code(nlayers, field_out, field_in, &
                               ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k
    real(kind=r_def), dimension(nlayers) :: partial
    real(kind=r_def), dimension(nlayers) :: swept
    partial(1) = field_in(map_w3(1))
    do k = 2, nlayers
      partial(k) = partial(k - 1) + field_in(map_w3(1) + k - 1)
    end do
    swept(nlayers) = partial(nlayers)
    do k = nlayers - 1, 1, -1
      swept(k) = swept(k + 1) - partial(k)
    end do
    do k = 1, nlayers
      field_out(map_w3(1) + k - 1) = swept(k)
    end do
  end subroutine column_solve_code
end module column_solve_kernel_mod
"""


# A kernel of the shape the hierarchical launch exists for: a scalar set once,
# a level loop whose iterations are independent, and a boundary element written
# outside it. The three become, in order, a plain assignment every team member
# makes, a TeamVectorRange the members share, and a write one member makes
# inside a Kokkos::single. It holds no automatic array, so it selects the
# hierarchical launch on the loop alone.
_LEVEL_ALGORITHM = """
program kokkos_level_test
  use field_mod, only : field_type
  use column_scale_kernel_mod, only : column_scale_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  call invoke(column_scale_kernel_type(out_field, in_field))
end program kokkos_level_test
"""


_LEVEL_KERNEL = """
module column_scale_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: column_scale_kernel_type
    type(arg_type) :: meta_args(2) = (/                              &
         arg_type(gh_field, gh_real, gh_write, w3),                  &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: column_scale_code
  end type column_scale_kernel_type
contains
  subroutine column_scale_code(nlayers, field_out, field_in, &
                               ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k
    real(kind=r_def) :: scaling
    scaling = 0.5_r_def
    do k = 1, nlayers - 1
      field_out(map_w3(1) + k - 1) = scaling * field_in(map_w3(1) + k - 1)
    end do
    field_out(map_w3(1) + nlayers - 1) = field_in(map_w3(1) + nlayers - 1)
  end subroutine column_scale_code
end module column_scale_kernel_mod
"""


# The same kernel with one array declared the way apply_helmholtz_operator's
# `coeff` is -- `dimension(max_length,4)` -- which is the commonest shape among
# the kernels the catalogue's `local-array` row blocks. A literal bound is not
# a Reference, so before extents were rendered rather than named this kernel
# was refused whole.
_LITERAL_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(nlayers,4) :: swept").replace(
    "    swept(nlayers) = partial(nlayers)",
    "    swept(nlayers,1) = partial(nlayers)").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k,1) = swept(k + 1,1) - partial(k)").replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = swept(k,1)")


# The shape the catalogue's `array-bound` row actually counts. `u_e`, `u_av`
# and `pert` are each declared `dimension(0:nlayers)` in the kernels that row
# blocks, so the View is one element longer than its upper bound and every
# subscript of it is shifted by nothing rather than by one.
_ZERO_BASED_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(0:nlayers) :: u_e").replace(
    "swept(", "u_e(")


# An origin that is not an integer expression over named sizes. The grammar is
# the extent's, and it is applied to the origin for the reason the two are
# read together: an origin the region cannot write shifts every subscript of
# the array rather than sizing it wrongly.
_UNRENDERABLE_ORIGIN_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept",
    "dimension(nlayers**nlayers:nlayers) :: swept")


# The statements convert_hdiv_native_code is built out of, with the routine it
# calls left out: a rank-2 local named whole and set to zero, accumulated into
# by column inside a loop over its second dimension, and a written field
# section taking one of its columns. Between them these are the three shapes
# the lowering has to reach, and the kernel family writes no fourth that does
# not need MATMUL.
_HDIV_SECTION_KERNEL = _SECTION_KERNEL.replace(
    "    integer(kind=i_def) :: nl, w3_idx, b_idx\n",
    "    integer(kind=i_def) :: i, nl, w3_idx, b_idx\n"
    "    real(kind=r_tran), dimension(nlayers, 3) :: vector\n").replace(
    "    difference(w3_idx : w3_idx + nl) = &\n"
    "        mass_flux(b_idx + 1 : b_idx + nl + 1) "
    "- mass_flux(b_idx : b_idx + nl)",
    "    vector = 0.0_r_tran\n"
    "    do i = 1, 3\n"
    "      vector(:,i) = vector(:,i) + mass_flux(b_idx : b_idx + nl)\n"
    "    end do\n"
    "    difference(w3_idx : w3_idx + nl) = vector(:,1)")


# A kernel module declaring genuine state beside its routine and making it
# public, which is how profile_interp_kernel_mod carries the profile it
# interpolates: a fixed-shape array and a scalar, set by the model before the
# invoke and read by the kernel. Neither is a constant and neither is an
# argument, so the region takes both as formals of its own.
_MODULE_VARIABLE_KERNEL = _LOCAL_KERNEL.replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  real(kind=r_def), public :: profile_heights(100)\n"
    "  integer(kind=i_def), public :: profile_size\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) * "
    "profile_heights(profile_size)")


# The same module variable declared `allocatable`. Its shape is not in the
# declaration at all, so there is no View for the region to size and nothing
# the generated interface could state.
_ALLOCATABLE_MODULE_VARIABLE_KERNEL = _MODULE_VARIABLE_KERNEL.replace(
    "  real(kind=r_def), public :: profile_heights(100)",
    "  real(kind=r_def), public, allocatable :: profile_heights(:)")


# dg_matrix_vector's shape: an LMA operator read as a rank-3 array, a leading
# cell argument LFRic supplies to every kernel that takes one, and the column
# addressed by array section on the operator's first dimension. That is what
# all four of the kernels this capability releases look like.
_OPERATOR_ALGORITHM = """
program kokkos_operator_test
  use field_mod,    only : field_type
  use operator_mod, only : operator_type
  use dg_matrix_vector_kernel_mod, only : dg_matrix_vector_kernel_type
  implicit none
  type(field_type)    :: lhs, x
  type(operator_type) :: matrix
  call invoke(dg_matrix_vector_kernel_type(lhs, x, matrix))
end program kokkos_operator_test
"""


_OPERATOR_KERNEL = """
module dg_matrix_vector_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_operator, gh_real,       &
                           gh_readwrite, gh_read, cell_column,             &
                           any_discontinuous_space_1, any_space_1
  use constants_mod, only : i_def, r_def
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: dg_matrix_vector_kernel_type
    type(arg_type) :: meta_args(3) = (/                                    &
         arg_type(gh_field,    gh_real, gh_readwrite,                      &
                                        any_discontinuous_space_1),        &
         arg_type(gh_field,    gh_real, gh_read, any_space_1),             &
         arg_type(gh_operator, gh_real, gh_read,                           &
                               any_discontinuous_space_1, any_space_1) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: dg_matrix_vector_code
  end type dg_matrix_vector_kernel_type
contains
  subroutine dg_matrix_vector_code(cell, nlayers, lhs, x, ncell_3d, matrix, &
                                   ndf1, undf1, map1, ndf2, undf2, map2)
    integer(kind=i_def), intent(in) :: cell, nlayers, ncell_3d
    integer(kind=i_def), intent(in) :: ndf1, undf1, ndf2, undf2
    integer(kind=i_def), dimension(ndf1), intent(in) :: map1
    integer(kind=i_def), dimension(ndf2), intent(in) :: map2
    real(kind=r_def), dimension(undf1), intent(inout) :: lhs
    real(kind=r_def), dimension(undf2), intent(in) :: x
    real(kind=r_def), dimension(ncell_3d,ndf1,ndf2), intent(in) :: matrix
    integer(kind=i_def) :: df, m, ik, i1, i2, nl
    nl = nlayers - 1
    ik = (cell - 1) * nlayers + 1
    do m = 1, ndf2
      i2 = map2(m)
      do df = 1, ndf1
        i1 = map1(df)
        lhs(i1:i1+nl) = lhs(i1:i1+nl) &
                      + matrix(ik:ik+nl, df, m) * x(i2:i2+nl)
      end do
    end do
  end subroutine dg_matrix_vector_code
end module dg_matrix_vector_kernel_mod
"""


# The kind of a module constant is stated only in its own module, so the
# transformation reads it there rather than guessing. A real run reaches it
# because generate() puts the kernel search path on the ModuleManager; these
# tests drive parse() and PSyFactory directly, so they say so themselves.
_PLANET_CONFIG = """
module planet_config_mod
  use constants_mod, only : i_def, l_def, r_def, r_quad
  implicit none
  real(kind=r_def), public, protected :: recip_epsilon = 1.0_r_def
  integer(kind=i_def), public, parameter :: n_moist = 3
  logical(l_def), public, protected :: rehabilitate = .false.
  logical, public, protected :: quenching = .true.
  real(kind=r_quad) :: unmapped_width
end module planet_config_mod
"""


def _invoke(tmp_path, name, algorithm_source, kernel_source, extra=None,
            dist_mem=True):
    """Build a one-invoke LFRic PSy layer from the given sources.

    :param tmp_path: the directory the sources are written to.
    :type tmp_path: :py:class:`pathlib.Path`
    :param str name: stem of the algorithm and kernel file names.
    :param str algorithm_source: the algorithm layer to parse.
    :param str kernel_source: the kernel module ``name`` resolves to.
    :param extra: further kernel modules, keyed by module name. An invoke
        calling two kernels from different modules needs a file per module,
        because the parser resolves each ``use`` by file name.
    :type extra: Optional[Dict[str, str]]
    :param bool dist_mem: whether to build with distributed memory. An LFRic
        invoke with it disabled and no kernel asking for a mesh property has
        no mesh object at all, which is the one state colouring cannot be
        completed from.

    :returns: the PSy layer, its first loop and that loop's first kernel.
    :rtype: Tuple[:py:class:`psyclone.psyGen.PSy`,
        :py:class:`psyclone.domain.lfric.LFRicLoop`,
        :py:class:`psyclone.domain.lfric.LFRicKern`]

    """
    Config.get().api = "lfric"
    algorithm = tmp_path / f"{name}_alg.f90"
    kernel = tmp_path / f"{name}_kernel_mod.f90"
    algorithm.write_text(algorithm_source, encoding="utf-8")
    kernel.write_text(kernel_source, encoding="utf-8")
    for module_name, module_source in (extra or {}).items():
        (tmp_path / f"{module_name}.f90").write_text(
            module_source, encoding="utf-8")
    (tmp_path / "planet_config_mod.f90").write_text(
        _PLANET_CONFIG, encoding="utf-8")
    # Assigned rather than appended to. Config is a singleton for the session
    # and psyclone.tests.utilities.get_invoke() appends an infrastructure path
    # to it without ever removing one, so a test that ran a GOcean invoke
    # earlier leaves external/dl_esm_inf/finite_difference/src here -- a
    # submodule that is not checked out, which get_kernel_filepath() then
    # refuses to search. That surfaces as the module constant failing to type
    # rather than as anything about the path, so these tests say what they
    # need rather than inheriting it.
    Config.get().include_paths = [str(tmp_path)]
    ModuleManager.get().add_search_path(str(tmp_path))
    _, invoke_info = parse(
        str(algorithm), api="lfric", kernel_paths=[str(tmp_path)])
    psy = PSyFactory(
        "lfric", distributed_memory=dist_mem).create(invoke_info)
    loop = psy.invokes.invoke_list[0].schedule.walk(LFRicLoop)[0]
    return psy, loop, loop.kernels()[0]


def _formals(code):
    """Return the names of a generated region's formals, in order.

    The pointer marker is dropped so that an array formal is named as its
    declaration names it, ``map_f_data`` rather than ``*map_f_data``.

    :param str code: the generated region.

    :returns: the region's formal parameter names, in the order declared.
    :rtype: List[str]

    """
    signature = code.split(") {\n")[0]
    return [parameter.strip().split()[-1].lstrip("*")
            for parameter in signature.split(",") if parameter.strip().split()]


def _coloured_inner(schedule):
    """Return the cells-in-colour loop of an already-coloured schedule.

    :param schedule: the invoke schedule 'LFRicColourTrans' has rewritten.
    :type schedule: :py:class:`psyclone.psyGen.InvokeSchedule`

    :returns: the inner loop, the one that runs the cells of one colour.
    :rtype: :py:class:`psyclone.domain.lfric.LFRicLoop`

    """
    return [loop for loop in schedule.walk(LFRicLoop)
            if loop.loop_type == "cells_in_colour"][0]


# A kernel that runs over the owned cells and the first halo cell together,
# writing a discontinuous space. Its loop begins at the first cell as a
# plain cell-column loop does and runs further, so the launch needs nothing
# but the count it already takes.
_OWNED_AND_HALO_ALGORITHM = """
program kokkos_owned_and_halo_test
  use field_mod, only : field_type
  use wide_write_kernel_mod, only : wide_write_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  integer :: hdepth
  call invoke(wide_write_kernel_type(out_field, in_field, hdepth))
end program kokkos_owned_and_halo_test
"""


_OWNED_AND_HALO_KERNEL = """
module wide_write_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           owned_and_halo_cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: wide_write_kernel_type
    type(arg_type) :: meta_args(2) = (/              &
         arg_type(gh_field, gh_real, gh_write, w3),  &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    integer :: operates_on = owned_and_halo_cell_column
  contains
    procedure, nopass :: wide_write_code
  end type wide_write_kernel_type
contains
  subroutine wide_write_code(nlayers, out_field, in_field, &
                             ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    real(kind=r_def), dimension(undf_w3), intent(inout) :: out_field
    real(kind=r_def), dimension(undf_w3), intent(in) :: in_field
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        out_field(map_w3(df) + k) = 2.0_r_def * in_field(map_w3(df) + k)
      end do
    end do
  end subroutine wide_write_code
end module wide_write_kernel_mod
"""


# The same kernel over the halo cells alone. Its loop begins where the owned
# cells end, which is the one iteration space that gives the launch a lower
# bound to carry.
_HALO_CELL_ALGORITHM = _OWNED_AND_HALO_ALGORITHM.replace(
    "kokkos_owned_and_halo_test", "kokkos_halo_cell_test").replace(
    "wide_write", "halo_write")


_HALO_CELL_KERNEL = _OWNED_AND_HALO_KERNEL.replace(
    "wide_write", "halo_write").replace(
    "owned_and_halo_cell_column", "halo_cell_column")


_FACE_QUADRATURE_ALGORITHM = """
program kokkos_face_quadrature_test
  use field_mod, only : field_type
  use quadrature_face_mod, only : quadrature_face_type
  use face_weight_kernel_mod, only : face_weight_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  type(quadrature_face_type) :: qr
  call invoke(face_weight_kernel_type(out_field, in_field, qr))
end program kokkos_face_quadrature_test
"""


# Face quadrature is a third shape: one point count, a face count, rank-2
# weights over the two of them and a basis array shaped (dim, ndf, np_xyz,
# nfaces). PSyclone's own generic names are used for the formals, which is
# what a stub-generated kernel declares and what the model's own face kernels
# name differently -- the region reads the names from the kernel either way.
_FACE_QUADRATURE_KERNEL = """
module face_weight_kernel_mod
  use argument_mod, only : arg_type, func_type, gh_field, gh_real, gh_write, &
                           gh_read, gh_basis, cell_column, gh_quadrature_face
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: face_weight_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_write, w3),                        &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    type(func_type) :: meta_funcs(1) = (/                                  &
         func_type(w3, gh_basis) /)
    integer :: gh_shape = gh_quadrature_face
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: face_weight_code
  end type face_weight_kernel_type
contains
  subroutine face_weight_code(nlayers, field_out, field_in,                &
                              ndf_w3, undf_w3, map_w3, basis_w3,           &
                              nfaces, np_xyz, weights_xyz)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: nfaces, np_xyz
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    real(kind=r_def), dimension(np_xyz,nfaces), intent(in) :: weights_xyz
    real(kind=r_def), dimension(1,ndf_w3,np_xyz,nfaces), intent(in) ::     &
                                                                 basis_w3
    integer(kind=i_def) :: k, df, qp, face
    real(kind=r_def) :: total
    do k = 0, nlayers - 1
      total = 0.0_r_def
      do df = 1, ndf_w3
        do face = 1, nfaces
          do qp = 1, np_xyz
            total = total + weights_xyz(qp,face)                           &
                  * basis_w3(1,df,qp,face) * field_in(map_w3(df) + k)
          end do
        end do
      end do
      field_out(map_w3(1) + k) = total
    end do
  end subroutine face_weight_code
end module face_weight_kernel_mod
"""


# ---------------------------------------------------------------------------
# Task E7: a TARGET dummy is inlinable.
# ---------------------------------------------------------------------------


# A helper whose read column carries `target`, which is the shape
# subgrid_vertical_support_mod's `third_order_vertical_edge` has:
# `real(kind=r_tran), target, intent(in) :: field(nlayers)`. `target` says
# only that a pointer may be aimed at the actual, which an inlined body
# neither creates nor breaks, so the declaration is unsupported without the
# routine being uninlinable.
_TARGET_DUMMY_KERNEL = _LOCAL_KERNEL.replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), target, intent(in) :: source(n)\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column\n"
    "end module column_solve_kernel_mod")


# The same helper with the column a pointer instead. A pointer dummy says
# something about storage that binding a formal to an actual does not
# reproduce, so it stays unsupported and `InlineTrans` refuses as before.
_POINTER_DUMMY_KERNEL = _TARGET_DUMMY_KERNEL.replace(
    "    real(kind=r_def), target, intent(in) :: source(n)\n",
    "    real(kind=r_def), pointer, intent(in) :: source(:)\n")


# The same helper in a module of its own, and a kernel that `use`s it. Two
# kernels of one invoke reach the very same parsed module, so this is the
# shape that would show a rewrite of the callee leaking from one capture into
# the next.
_TARGET_HELPER_MODULE = """
module target_helper_mod
  use constants_mod, only : i_def, r_def
  implicit none
  public :: sweep_column
contains
  subroutine sweep_column(n, source, result)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), target, intent(in) :: source(n)
    real(kind=r_def), dimension(n), intent(inout) :: result
    integer(kind=i_def) :: j
    result(n) = source(n)
    do j = n - 1, 1, -1
      result(j) = result(j + 1) - source(j)
    end do
  end subroutine sweep_column
end module target_helper_mod
"""


_TARGET_CALLEE_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use target_helper_mod, only : sweep_column").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n")


# The second kernel of the pair, differing from the first only in its name.
_TARGET_TWIN_KERNEL = _TARGET_CALLEE_KERNEL.replace(
    "column_solve", "column_twin")


_TARGET_TWIN_ALGORITHM = """
program kokkos_target_twin_test
  use field_mod, only : field_type
  use column_solve_kernel_mod, only : column_solve_kernel_type
  use column_twin_kernel_mod, only : column_twin_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  call invoke(column_solve_kernel_type(out_field, in_field), &
              column_twin_kernel_type(out_field, in_field))
end program kokkos_target_twin_test
"""


# A module holding nothing but a named constant, and a second module holding a
# pure function that reads it. Two containers deep is the shape
# `face_from_face_selector` has -- a function of
# `sci_face_selector_support_mod` reading `W`, `S`, `E` and `N` from
# `reference_element_mod` -- and it is the
# shape that makes bringing the callee in worth doing: the function's body can
# move into the kernel's Container, and the constant it reads moves with it.
_EDGE_INDEX_MODULE = """
module edge_index_mod
  use constants_mod, only : i_def
  implicit none
  private
  integer(kind=i_def), public, parameter :: top_edge = 7_i_def
end module edge_index_mod
"""


_COLUMN_SELECT_MODULE = """
module column_select_mod
  use constants_mod, only : i_def
  use edge_index_mod, only : top_edge
  implicit none
  private
  public :: selected_level
contains
  pure function selected_level(n) result(level)
    integer(kind=i_def), intent(in) :: n
    integer(kind=i_def) :: level
    level = n + top_edge
  end function selected_level
end module column_select_mod
"""


# The kernel that reads it. `selected_level(k)` stands in an expression, and
# the kernel's own file does not say whether that is a function reference or an
# array element, so the frontend leaves a Call behind and the symbol at the
# call site an unspecialised `Symbol`. That is what the capture has to see
# through.
_USED_FUNCTION_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use column_select_mod, only : selected_level").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(selected_level(k) - 7)")


# --------------------------------------------------------------------------
# Task E8: a callee that calls, or reads, its own module.
# --------------------------------------------------------------------------


# A helper module holding two procedures, the public one calling the private
# one. This is `panel_edge_support_mod`'s shape: `crosses_panel_edge` calls
# `rotated_panel_neighbour`, both of that module, and the kernel `use`s only
# the first. The sibling is called twice, and reads a constant the module --
# not the sibling -- imports, which is what makes the second inlining of it a
# different problem from the first.
_SIBLING_CALLEE_MODULE = """
module sweep_support_mod
  use constants_mod, only : i_def, r_def
  use edge_index_mod, only : top_edge
  implicit none
  private
  public :: sweep_column
contains
  subroutine sweep_column(n, source, result)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    integer(kind=i_def) :: j
    result(n) = source(n) * damping(n)
    do j = n - 1, 1, -1
      result(j) = result(j + 1) - source(j) * damping(j)
    end do
  end subroutine sweep_column
  function damping(level) result(factor)
    integer(kind=i_def), intent(in) :: level
    real(kind=r_def) :: factor
    factor = 1.0_r_def / real(level + top_edge, r_def)
  end function damping
end module sweep_support_mod
"""


# The same module with the sibling reading a variable of it rather than a
# constant of a third. This is `sci_chi_transform_mod`'s shape, where
# `chi2xyz` reads `chi2xyz_rot_mat`: the sibling call is reachable and the
# datum is not, so which of the two the refusal names is the whole question.
_SIBLING_STATE_MODULE = _SIBLING_CALLEE_MODULE.replace(
    "  use edge_index_mod, only : top_edge\n",
    "").replace(
    "  private\n",
    "  private\n"
    "  real(kind=r_def) :: relaxation = 0.5_r_def\n").replace(
    "    factor = 1.0_r_def / real(level + top_edge, r_def)",
    "    factor = relaxation / real(level, r_def)")


# The kernel that calls the public procedure of either module. Written the
# same way as _EXTERNAL_CALLEE_KERNEL, so that the only difference between
# the two is whether the callee calls anything itself.
_SIBLING_CALLEE_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use sweep_support_mod, only : sweep_column").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n")


# The same module with the sibling declaring a local that has an initialiser,
# which Fortran gives the SAVE attribute and InlineTrans will not inline. The
# call to it therefore stays where the file put it, so that what happens to a
# sibling that cannot be absorbed can be asked.
_STATIC_SIBLING_MODULE = _SIBLING_CALLEE_MODULE.replace(
    "    integer(kind=i_def), intent(in) :: level\n",
    "    integer(kind=i_def), intent(in) :: level\n"
    "    integer(kind=i_def) :: seen = 0_i_def\n").replace(
    "    factor = 1.0_r_def / real(level + top_edge, r_def)",
    "    seen = seen + 1_i_def\n"
    "    factor = 1.0_r_def / real(level + top_edge + seen, r_def)")
