# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the deliberately narrow LFRic-to-Kokkos transformation."""

# pylint: disable=protected-access

import pytest

from psyclone.configuration import Config
from psyclone.core import AccessType
from psyclone.domain.lfric import LFRicLoop
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.lfric import LFRicArgStencil
from psyclone.parse import ModuleManager
from psyclone.parse.algorithm import parse
from psyclone.psyGen import PSyFactory
from psyclone.psyir.backend.kokkos import KokkosRegion, KokkosScalar
from psyclone.psyir.nodes import (
    ArrayReference, CodeBlock, IntrinsicCall, Literal, Range)
from psyclone.psyir.symbols import (
    ContainerSymbol, ImportInterface, ScalarType, Symbol)
from psyclone.psyir.transformations import TransformationError


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


# The same kernel twice in one invoke, so that the captured loop has a sibling
# still counting with the PSy layer's cell counter.
_PAIRED_ALGORITHM = """
program kokkos_paired_test
  use constants_mod, only : r_tran
  use field_mod, only : field_type
  use scaled_copy_kernel_mod, only : scaled_copy_kernel_type
  implicit none
  type(field_type) :: out_field, other_field, in_field
  real(kind=r_tran) :: scaling
  call invoke(scaled_copy_kernel_type(out_field, in_field, scaling),   &
              scaled_copy_kernel_type(other_field, in_field, scaling))
end program kokkos_paired_test
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


# The same kernel with the column floored at a constant imported from a module
# that is not on the search path, so its symbol stays unresolved. The lowering
# has to expand every reference in the statement to decide whether it is an
# array, and cannot do that for a symbol whose declaration it has never seen.
# The argument of an intrinsic is where the two answers part company:
# ArrayAssignment2LoopsTrans.validate() skips references a Call encloses, and
# its apply() expands them like any other. A kernel of this shape, in
# solver_moist_correction_alg_mod, is what showed the difference.
_UNRESOLVED_SECTION_KERNEL = _SECTION_KERNEL.replace(
    "  use fs_continuity_mod, only : w3, w2v",
    "  use fs_continuity_mod, only : w3, w2v\n"
    "  use unresolvable_constants_mod, only : eps").replace(
    "difference(w3_idx : w3_idx + nl) = &",
    "difference(w3_idx : w3_idx + nl) = max(eps, &").replace(
    "- mass_flux(b_idx : b_idx + nl)",
    "- mass_flux(b_idx : b_idx + nl))")


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


# The same shape with the scalar made logical. Stage 2 refused this, because
# LFRic's l_def is kind(.false.) and measures 4 bytes where PSyclone's
# precision map records it as 1 (issue #1941), so a logical(c_bool) dummy would
# have sat against a logical(4) actual. Stage 5 admits it by conversion
# instead: the dummy is logical(c_bool), value and the call site wraps the
# actual in LOGICAL(..., c_bool), which is correct at either width. Neither
# side reads the precision map for it, so #1941 is bypassed rather than
# depended on.
_LOGICAL_ALGORITHM = """
program kokkos_logical_test
  use constants_mod, only : l_def
  use r_solver_field_mod, only : r_solver_field_type
  use masked_solver_kernel_mod, only : masked_solver_kernel_type
  implicit none
  type(r_solver_field_type) :: out_field, in_field
  logical(kind=l_def) :: masked
  call invoke(masked_solver_kernel_type(out_field, in_field, masked))
end program kokkos_logical_test
"""


_LOGICAL_KERNEL = """
module masked_solver_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_scalar, gh_real, &
                           gh_logical, gh_write, gh_read, cell_column
  use constants_mod, only : i_def, l_def, r_solver
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: masked_solver_kernel_type
    type(arg_type) :: meta_args(3) = (/                              &
         arg_type(gh_field,  gh_real,    gh_write, w3),              &
         arg_type(gh_field,  gh_real,    gh_read,  w3),              &
         arg_type(gh_scalar, gh_logical, gh_read) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: masked_solver_code
  end type masked_solver_kernel_type
contains
  subroutine masked_solver_code(nlayers, field_out, field_in, masked, &
                                ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    real(kind=r_solver), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_solver), dimension(undf_w3), intent(in) :: field_in
    logical(kind=l_def), intent(in) :: masked
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        if (masked) then
          field_out(map_w3(df) + k) = field_in(map_w3(df) + k)
        end if
      end do
    end do
  end subroutine masked_solver_code
end module masked_solver_kernel_mod
"""


# The same kernel with the logical made an array. A scalar crosses the ABI by
# conversion, value by value, which is what makes the two widths irrelevant.
# An array crosses by reference: a View<bool*> laid over logical(l_def) storage
# reinterprets 4-byte elements as 1-byte ones and reads every fourth byte, so
# the conversion that fixes the scalar has nothing to act on. The metadata is
# unchanged -- gh_scalar/gh_logical -- because it is the Fortran declaration
# that makes it an array; a real LFRic kernel could not declare it this way,
# and the refusal is checked from the declaration rather than the metadata.
_LOGICAL_ARRAY_KERNEL = _LOGICAL_KERNEL.replace(
    "    logical(kind=l_def), intent(in) :: masked",
    "    logical(kind=l_def), dimension(ndf_w3), intent(in) :: masked"
    ).replace("        if (masked) then", "        if (masked(df)) then")


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


# The same kernel through a 1-D CROSS stencil, which is the shape the
# transformation refuses. The refusal is not a matter of taste: LFRic gives a
# 1-D stencil's size to the kernel as a *scalar* formal, fed per cell from
# 'field_in_stencil_size(cell)', where CROSS2D gives an array formal fed from
# a whole array. 'apply''s per-cell rule appends the cell index to an array
# actual, so the array form needs nothing new and the scalar form would need
# a per-cell scalar argument kind that does not exist. The dofmap loses its
# branch dimension for the same reason. Both differences are visible below.
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


# The same kernel with a scalar local named after one of the identifiers the
# team launch declares around the kernel body. The kernel's declaration would
# shadow the launch's and then be assigned to, which C++ accepts: the region
# would run with a team size the kernel had overwritten. That is a wrong answer
# rather than a compile error, which is why it is refused.
_TEAM_NAME_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k, team_size\n").replace(
    "    partial(1) = field_in(map_w3(1))",
    "    team_size = nlayers\n    partial(1) = field_in(map_w3(1))").replace(
    "    do k = 2, nlayers", "    do k = 2, team_size")


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


# The same kernel with a scalar local named 'team' and no automatic array. The
# hierarchical launch declares 'team' as its lambda parameter whether or not
# there is scratch to place, so the reserved names cannot be conditional on
# there being an automatic array as they were until this stage.
_TEAM_LEVEL_KERNEL = _LEVEL_KERNEL.replace(
    "    real(kind=r_def) :: scaling",
    "    integer(kind=i_def) :: team\n"
    "    real(kind=r_def) :: scaling").replace(
    "    scaling = 0.5_r_def",
    "    team = nlayers\n    scaling = 0.5_r_def").replace(
    "    do k = 1, nlayers - 1", "    do k = 1, team - 1")


# A kernel with no local arrays and a scalar local named 'ncells'. The cell
# count is declared by both launch shapes, not only the team one, so this
# refusal does not depend on there being scratch to place.
_NCELLS_LOCAL_KERNEL = _KERNEL.replace(
    "integer(kind=i_def) :: k, df",
    "integer(kind=i_def) :: k, df, ncells").replace(
    "    do k = 0, nlayers - 1",
    "    ncells = nlayers\n    do k = 0, ncells - 1")


# The same kernel with one array sized by a module constant instead of by a
# formal. Fortran allows it -- a module entity is as valid an automatic bound
# as a dummy -- but the launch computes its scratch size before it enters the
# region, so an extent it cannot name there cannot be sized.
_UNSIZED_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : n_moist").replace(
    "dimension(nlayers) :: swept", "dimension(n_moist) :: swept")


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


# The same kernel with a lower bound that is not 1. KokkosView and
# KokkosScratch subtract a fixed 1 from each Fortran index, so this cannot be
# described without index offsets becoming expressions.
_LOWER_BOUND_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(0:nlayers-1) :: swept")


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


# SIZE in a dimension other than the first, of a rank-2 local. The literal
# bound is the answer, so the substitution has to reach the second entry of
# the declared shape rather than assuming the first.
_RANK_TWO_BOUND_KERNEL = _LITERAL_LOCAL_KERNEL.replace(
    "      swept(k,1) = swept(k + 1,1) - partial(k)",
    "      swept(k,1) = swept(k + 1,1) - partial(k) * size(swept, 2)")


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


# UBOUND of an array whose lower bound is not 1. The extent grammar already
# refuses that declaration, and the enquiry inherits the refusal rather than
# paraphrasing it, because the answer depends on the same bounds.
_LOWER_BOUND_ENQUIRY_KERNEL = _LOWER_BOUND_LOCAL_KERNEL.replace(
    "    swept(nlayers) = partial(nlayers)",
    "    swept(ubound(swept, 1)) = partial(nlayers)")


# A whole-array assignment, which is where these enquiries mostly come from:
# the lowering writes LBOUND and UBOUND into the loop bounds of every
# full-extent section it rewrites, so they are produced by the transformation
# rather than by the kernel author.
_FULL_SECTION_KERNEL = _SECTION_KERNEL.replace(
    "    difference(w3_idx : w3_idx + nl) = &\n"
    "        mass_flux(b_idx + 1 : b_idx + nl + 1) "
    "- mass_flux(b_idx : b_idx + nl)",
    "    difference(:) = difference(:) + (b_idx + nl)")


# The same kernel declaring its own constants beside the routine, the way
# poly1d_reconstruction and create_w2mask do. The module is `private`, so the
# PSy layer could not import either name even if it wanted to; the value is
# what reaches the region.
_STATIC_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  integer(kind=i_def), parameter :: nfaces = 4\n"
    "  real(kind=r_def), parameter :: tol = 1.0e-9_r_def\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) * nfaces + tol")


# The same, with the constant declared as an array. `x_dofs(2) = (/ 1, 3 /)`
# is what fractional_horizontal_wind writes, and there is no single literal to
# substitute for a reference into it.
_ARRAY_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  integer(kind=i_def), parameter :: x_dofs(2) = (/ 1, 3 /)\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(x_dofs(1))")


# A kernel spelling a kind in its body rather than only in its declarations.
# `real(x, r_def)` puts r_def into the tree as a Reference, which is not data
# the PSy layer passes by value: the cast consumes it.
_CAST_KIND_KERNEL = _LOCAL_KERNEL.replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = swept(k) + real(k, r_def)")


# The same, casting to a kind no declaration in the body repeats. The width
# has to come from the cast argument or the region silently computes at the C
# writer's default instead of at the width the Fortran asked for.
_UNDECLARED_CAST_KIND_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, r_def, r_second").replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = swept(k) + real(k, r_second)")


# A kernel importing a logical constant. It was named for being off the ABI,
# which stage 5 made false: a logical scalar now crosses by conversion, and an
# imported constant crosses as an argument rather than a literal because its
# value is known only where the PSy layer runs. The name is kept so that the
# fixture's history is legible against the plans that refer to it.
_OFF_ABI_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : rehabilitate").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      if (rehabilitate) swept(k) = swept(k + 1) - partial(k)")


# A kernel importing a module datum whose kind the ABI does not carry.
# 'unmapped_width' is an r_quad real, 16 bytes and so off a C ABI carrying 4-
# and 8-byte ones, and it is declared with no attributes so that PSyIR models
# it rather
# than leaving the text to be re-read. Stage 5 needs this because the case used
# to be carried by an l_def logical constant, which is now admitted: the
# refusal is unchanged, so it keeps a witness.
_UNMAPPED_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : unmapped_width").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) * unmapped_width")


# A kernel importing a constant from a module PSyclone cannot read. Its kind
# is stated only there, so there is no width to put on the ABI and no honest
# guess to make -- the largest single cause left in the survey's residue.
_UNREADABLE_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use unreadable_constants_mod, only : eps").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) + eps")


# The same, casting to a kind the LFRic precision map does not carry. The map
# names the kinds the model computes in; `r_native` is the compiler's own
# default, and there is no width to record for it.
_UNMAPPED_CAST_KIND_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, r_def, r_native").replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = real(swept(k), r_native)")


# A kernel calling a function from a module PSyclone has not read. Without the
# module the frontend cannot tell `helper(k)` from an array reference, so the
# symbol is a plain Symbol until something resolves it -- which is why the
# refusal has to be the one about calls rather than one about module data.
_CALLED_ROUTINE_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use helper_mod, only : helper").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - helper(k)")


# The module the kernel above calls into, written out only where a test needs
# PSyclone to have read it. Until it does, `helper` is a plain Symbol; once it
# has, `resolve_type` specialises it to a RoutineSymbol, which is the whole
# difference the refusal turns on.
_HELPER_MODULE = """
module helper_mod
  use constants_mod, only : r_def
  implicit none
contains
  function helper(i) result(scaled)
    integer, intent(in) :: i
    real(kind=r_def) :: scaled
    scaled = real(i, r_def)
  end function helper
end module helper_mod
"""


# A kernel module declaring a variable rather than a constant beside the
# routine. It looks like the constant case at the point the walk meets it --
# module scope, a literal beside the name -- and is not one: without
# `parameter` the value is an initialisation the module may overwrite, so
# writing it into the region would capture a state rather than a constant.
_STATIC_VARIABLE_KERNEL = _LOCAL_KERNEL.replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  real(kind=r_def) :: cached_tol = 1.0e-9_r_def\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) + cached_tol")


# The target kernel reading its one imported constant twice. Each reference is
# a separate node, and describing the second would re-resolve a symbol already
# on the ABI and offer the PSy layer the same argument twice.
_REPEATED_IMPORT_KERNEL = _KERNEL.replace(
    "            1.0_r_def + recip_epsilon * mr_v_at_dof",
    "            recip_epsilon + recip_epsilon * mr_v_at_dof")


# A kind-polymorphic kernel: one metadata name over several implementations
# that differ only in the precision of their real arguments. Built from a
# template rather than written out three times because the three fixtures below
# differ only in which two kinds the interface carries and which kind the
# algorithm passes, and that difference is the whole point of each test.
#
# The layout is sci_tri_solve_kernel_mod's: a public generic interface beside
# the metadata type, with the specific procedures named for their kind.
_POLYMORPHIC_MEMBER = """
  subroutine {name}_code_{kind}(nlayers, field_out, field_in, scaling, &
                                ndf_w3, undf_w3, map_w3)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    real(kind={kind}), dimension(undf_w3), intent(inout) :: field_out
    real(kind={kind}), dimension(undf_w3), intent(in) :: field_in
    real(kind={kind}), intent(in) :: scaling
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        field_out(map_w3(df) + k) = scaling * field_in(map_w3(df) + k)
      end do
    end do
  end subroutine {name}_code_{kind}
"""


def _polymorphic_kernel(name, first, second, stencil=False):
    """Build a kernel module whose code is an interface over two kinds.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str first: the kind of the first specific procedure.
    :param str second: the kind of the second.
    :param bool stencil: whether the read field carries a cross stencil. It
        changes nothing about the precisions; it puts the metadata outside
        what ``KernelInterface`` builds, which is a separate refusal.

    :returns: Fortran source for the module.
    :rtype: str
    """
    members = "".join(
        _POLYMORPHIC_MEMBER.format(name=name, kind=kind)
        for kind in (first, second))
    imports = "gh_read, cell_column"
    read_arg = "arg_type(gh_field,  gh_real, gh_read,  w3)"
    if stencil:
        imports = "gh_read, cell_column, stencil, cross"
        read_arg = "arg_type(gh_field,  gh_real, gh_read,  w3, stencil(cross))"
    return f"""
module {name}_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_scalar, gh_real, gh_write, &
                           {imports}
  use constants_mod, only : i_def, r_def, r_single, r_double, r_solver, r_quad
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: {name}_kernel_type
    type(arg_type) :: meta_args(3) = (/                              &
         arg_type(gh_field,  gh_real, gh_write, w3),                 &
         {read_arg},                 &
         arg_type(gh_scalar, gh_real, gh_read) /)
    integer :: operates_on = cell_column
  end type {name}_kernel_type
  public :: {name}_code
  interface {name}_code
    module procedure {name}_code_{first}, {name}_code_{second}
  end interface
contains
{members}end module {name}_kernel_mod
"""


def _polymorphic_algorithm(name, field_module, field_type, kind,
                           stencil=False):
    """Build an algorithm invoking one polymorphic kernel.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str field_module: the module the field type comes from.
    :param str field_type: the LFRic field type, whose precision selects the
        specific procedure.
    :param str kind: the kind of the scalar argument.
    :param bool stencil: whether the kernel's read field carries a stencil, in
        which case the invoke passes its extent as well.

    :returns: Fortran source for the program.
    :rtype: str
    """
    extent_declaration = ""
    extent_actual = ""
    if stencil:
        extent_declaration = "\n  integer :: extent = 1"
        extent_actual = ", extent"
    return f"""
program kokkos_{name}_test
  use constants_mod, only : {kind}
  use {field_module}, only : {field_type}
  use {name}_kernel_mod, only : {name}_kernel_type
  implicit none
  type({field_type}) :: out_field, in_field
  real(kind={kind}) :: scaling{extent_declaration}
  call invoke({name}_kernel_type(out_field, in_field{extent_actual}, scaling))
end program kokkos_{name}_test
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
  logical(kind=l_def), public, parameter :: rehabilitate = .false.
  real(kind=r_quad) :: unmapped_width
end module planet_config_mod
"""


def _invoke(tmp_path, name, algorithm_source, kernel_source):
    """Build a one-invoke LFRic PSy layer from the given sources."""
    Config.get().api = "lfric"
    algorithm = tmp_path / f"{name}_alg.f90"
    kernel = tmp_path / f"{name}_kernel_mod.f90"
    algorithm.write_text(algorithm_source, encoding="utf-8")
    kernel.write_text(kernel_source, encoding="utf-8")
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
    psy = PSyFactory("lfric", distributed_memory=True).create(invoke_info)
    loop = psy.invokes.invoke_list[0].schedule.walk(LFRicLoop)[0]
    return psy, loop, loop.kernels()[0]


@pytest.fixture(name="target")
# pylint: disable-next=unused-argument
def target_fixture(tmp_path, clear_module_manager_instance):
    """Create the production metadata/body in a minimal LFRic invoke."""
    return _invoke(tmp_path, "moist_dyn_gas", _ALGORITHM, _KERNEL)


@pytest.fixture(name="cell_local_target")
# pylint: disable-next=unused-argument
def cell_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares 'cell' as a local of its own."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _CELL_LOCAL_KERNEL)


@pytest.fixture(name="second_target")
# pylint: disable-next=unused-argument
def second_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an unrelated supported kernel in a minimal LFRic invoke."""
    return _invoke(
        tmp_path, "scaled_copy", _SECOND_ALGORITHM, _SECOND_KERNEL)


@pytest.fixture(name="halo_target")
# pylint: disable-next=unused-argument
def halo_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose loop is preceded by a halo exchange."""
    return _invoke(
        tmp_path, "halo_read", _HALO_ALGORITHM, _HALO_KERNEL)


@pytest.fixture(name="paired_target")
# pylint: disable-next=unused-argument
def paired_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose captured loop has a cell loop beside it."""
    return _invoke(
        tmp_path, "scaled_copy", _PAIRED_ALGORITHM, _SECOND_KERNEL)


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


@pytest.fixture(name="logical_target")
# pylint: disable-next=unused-argument
def logical_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel takes an l_def logical scalar."""
    return _invoke(
        tmp_path, "masked_solver", _LOGICAL_ALGORITHM, _LOGICAL_KERNEL)


@pytest.fixture(name="logical_array_target")
# pylint: disable-next=unused-argument
def logical_array_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel takes an l_def logical array."""
    return _invoke(
        tmp_path, "masked_solver", _LOGICAL_ALGORITHM, _LOGICAL_ARRAY_KERNEL)


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


@pytest.fixture(name="local_target")
# pylint: disable-next=unused-argument
def local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel holds two automatic column arrays."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _LOCAL_KERNEL)


@pytest.fixture(name="team_name_target")
# pylint: disable-next=unused-argument
def team_name_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local named 'team_size'."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _TEAM_NAME_KERNEL)


@pytest.fixture(name="tri_solve_target")
# pylint: disable-next=unused-argument
def tri_solve_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel is two recurrences and nothing else."""
    return _invoke(
        tmp_path, "tri_sweep", _TRI_SOLVE_ALGORITHM, _TRI_SOLVE_KERNEL)


@pytest.fixture(name="level_target")
# pylint: disable-next=unused-argument
def level_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel holds one parallelisable level loop."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _LEVEL_KERNEL)


@pytest.fixture(name="nested_level_target")
# pylint: disable-next=unused-argument
def nested_level_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel nests one level loop inside another."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _NESTED_LEVEL_KERNEL)


@pytest.fixture(name="stepped_level_target")
# pylint: disable-next=unused-argument
def stepped_level_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose level loop counts in twos."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _STEPPED_LEVEL_KERNEL)


@pytest.fixture(name="team_level_target")
# pylint: disable-next=unused-argument
def team_level_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local named 'team'."""
    return _invoke(
        tmp_path, "column_scale", _LEVEL_ALGORITHM, _TEAM_LEVEL_KERNEL)


@pytest.fixture(name="ncells_local_target")
# pylint: disable-next=unused-argument
def ncells_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local named 'ncells'."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _NCELLS_LOCAL_KERNEL)


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


@pytest.fixture(name="rank_two_bound_target")
# pylint: disable-next=unused-argument
def rank_two_bound_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke asking the SIZE of a rank-2 local's second axis."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _RANK_TWO_BOUND_KERNEL)


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


@pytest.fixture(name="lower_bound_enquiry_target")
# pylint: disable-next=unused-argument
def lower_bound_enquiry_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create an invoke asking the UBOUND of an array based other than at 1."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _LOWER_BOUND_ENQUIRY_KERNEL)


@pytest.fixture(name="full_section_target")
# pylint: disable-next=unused-argument
def full_section_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel assigns a whole array at once."""
    return _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM, _FULL_SECTION_KERNEL)


@pytest.fixture(name="static_constant_target")
# pylint: disable-next=unused-argument
def static_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel module declares its own constants."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _STATIC_CONSTANT_KERNEL)


@pytest.fixture(name="array_constant_target")
# pylint: disable-next=unused-argument
def array_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel module declares an array parameter."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ARRAY_CONSTANT_KERNEL)


@pytest.fixture(name="cast_kind_target")
# pylint: disable-next=unused-argument
def cast_kind_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel body names a kind in a cast."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _CAST_KIND_KERNEL)


@pytest.fixture(name="undeclared_cast_kind_target")
# pylint: disable-next=unused-argument
def undeclared_cast_kind_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke casting to a kind no declaration in the body uses."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _UNDECLARED_CAST_KIND_KERNEL)


@pytest.fixture(name="called_routine_target")
# pylint: disable-next=unused-argument
def called_routine_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel calls an unresolvable function."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _CALLED_ROUTINE_KERNEL)


@pytest.fixture(name="off_abi_constant_target")
# pylint: disable-next=unused-argument
def off_abi_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke importing a constant of a kind off the C ABI."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _OFF_ABI_CONSTANT_KERNEL)


@pytest.fixture(name="unmapped_constant_target")
# pylint: disable-next=unused-argument
def unmapped_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke importing a datum of a kind the ABI does not map."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _UNMAPPED_CONSTANT_KERNEL)


@pytest.fixture(name="unreadable_constant_target")
# pylint: disable-next=unused-argument
def unreadable_constant_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke importing a constant from an unreadable module."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _UNREADABLE_CONSTANT_KERNEL)


@pytest.fixture(name="unmapped_cast_kind_target")
# pylint: disable-next=unused-argument
def unmapped_cast_kind_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke casting to a kind the precision map does not carry."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _UNMAPPED_CAST_KIND_KERNEL)


@pytest.fixture(name="static_variable_target")
# pylint: disable-next=unused-argument
def static_variable_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel module declares a module variable."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _STATIC_VARIABLE_KERNEL)


@pytest.fixture(name="repeated_import_target")
# pylint: disable-next=unused-argument
def repeated_import_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads one imported constant twice."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _REPEATED_IMPORT_KERNEL)


@pytest.fixture(name="polymorphic_target")
# pylint: disable-next=unused-argument
def polymorphic_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke of an interface exactly one member of which matches.

    The r_double member is declared first, so a selection that returned
    ``schedules[0]`` would return the wrong one and every assertion about the
    generated types would fail. The algorithm passes r_solver fields, which
    ``precision_map`` gives 4 bytes, so the r_single member is the match.
    """
    return _invoke(
        tmp_path, "tri_scale",
        _polymorphic_algorithm(
            "tri_scale", "r_solver_field_mod", "r_solver_field_type",
            "r_solver"),
        _polymorphic_kernel("tri_scale", "r_double", "r_single"))


@pytest.fixture(name="unmatched_target")
# pylint: disable-next=unused-argument
def unmatched_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke of an interface no member of which matches.

    r_single is 4 bytes and r_quad is 16; the algorithm's ``field_type`` is
    r_def at 8. No monkeypatching: the widths are ``psyclone.cfg``'s.
    """
    return _invoke(
        tmp_path, "quad_scale",
        _polymorphic_algorithm(
            "quad_scale", "field_mod", "field_type", "r_def"),
        _polymorphic_kernel("quad_scale", "r_single", "r_quad"))


@pytest.fixture(name="ambiguous_target")
# pylint: disable-next=unused-argument
def ambiguous_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke of an interface every member of which matches.

    ``precision_map`` gives r_single and r_solver the same 4 bytes, so a
    matcher working in widths cannot separate them. Fortran can, because it
    resolves by name.
    """
    return _invoke(
        tmp_path, "dual_scale",
        _polymorphic_algorithm(
            "dual_scale", "r_solver_field_mod", "r_solver_field_type",
            "r_solver"),
        _polymorphic_kernel("dual_scale", "r_single", "r_solver"))


@pytest.fixture(name="unmodelled_target")
# pylint: disable-next=unused-argument
def unmodelled_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke of an interface the matcher cannot be asked about.

    The precisions are the matching fixture's, so nothing about the kinds
    stops selection. What stops it is the stencil: ``KernelInterface`` builds
    the interface the metadata implies before comparing anything, and issue
    #928 leaves stencils among the parts it does not build.
    """
    return _invoke(
        tmp_path, "stencil_scale",
        _polymorphic_algorithm(
            "stencil_scale", "r_solver_field_mod", "r_solver_field_type",
            "r_solver", stencil=True),
        _polymorphic_kernel("stencil_scale", "r_double", "r_single",
                            stencil=True))


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


def test_lfric_kokkos_trans_lowers_no_exchange_outside_an_invoke(target):
    """A loop with no invoke schedule above it is left alone.

    The exchanges are reached through the loop's ``InvokeSchedule`` ancestor,
    and a detached loop has none. Returning rather than walking from ``None``
    is what lets the helper be called on a loop held outside the tree it was
    parsed into, as a unit test does.
    """
    _, loop, _ = target

    assert LFRicKokkosTrans._lower_halo_exchanges(loop.detach()) is None


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


def test_lfric_kokkos_trans_refuses_an_unlowerable_section(
        tmp_path, clear_module_manager_instance):
    """A section the lowering will not touch stays a refusal.

    Widening a capture is only safe if the thing it delegates to keeps its
    own refusals, so this asserts that PSyclone's reason is carried out
    rather than swallowed.
    """
    # pylint: disable=unused-argument
    # pylint: disable-next=unused-variable
    _, loop, _ = _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM,
        _DEPENDENT_SECTION_KERNEL)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "cannot lower an array section to a loop" in str(error.value)
    assert "loop-carried dependencies" in str(error.value)


def test_lfric_kokkos_trans_section_check_predicts_the_lowering(
        tmp_path, clear_module_manager_instance):
    """The section check answers for the lowering, not for its validate.

    ArrayAssignment2LoopsTrans.validate() accepts this assignment and its
    apply() then refuses it: validate() skips the references a Call encloses
    and apply() expands them like any other. Asking the weaker question left
    _validate_sections() promising a prediction it did not make, so a caller
    using the two helpers in turn -- as the coverage survey does -- got an
    exception out of the lowering instead of a refusal.

    Asserted through the public validate(), which reports the section reason
    because the section check is complete on its own. Other checks would
    refuse this kernel too, for its unresolved import; that they run later is
    what makes the message evidence about this one.
    """
    # pylint: disable=unused-argument
    # pylint: disable-next=unused-variable
    _, loop, _ = _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM,
        _UNRESOLVED_SECTION_KERNEL)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "cannot lower an array section to a loop" in str(error.value)
    assert "eps" in str(error.value)


def test_lfric_kokkos_trans_refuses_a_section_outside_an_assignment(
        section_target):
    """Only a whole-column assignment is within reach of the lowering.

    An intrinsic called as a statement can hold a section that no assignment
    encloses, which the lowering has no way to rewrite. Validation says so
    itself rather than letting the backend fail later on a bare Range.
    """
    _, loop, kernel = section_target
    schedule = kernel.get_callees()[0]
    field = schedule.symbol_table.lookup("difference")
    integer = ScalarType.integer_type()
    section = ArrayReference.create(field, [Range.create(
        Literal("1", integer), Literal("2", integer))])
    schedule.addchild(IntrinsicCall.create(
        IntrinsicCall.Intrinsic.RANDOM_NUMBER, [section]))

    with pytest.raises(TransformationError, match="outside an assignment"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_validate_leaves_sections_alone(section_target):
    """Validation predicts the lowering; it must not perform it.

    validate() is public and callable on its own, so a caller deciding
    whether a loop is capturable would otherwise silently rewrite the
    kernel it was only asking about.
    """
    _, loop, kernel = section_target
    schedule = kernel.get_callees()[0]

    LFRicKokkosTrans().validate(loop)

    assert len(schedule.walk(Range)) == 3


def test_lfric_kokkos_trans_splices_one_loop(target):
    """The selected loop becomes one typed call and retains halo updates."""
    psy, loop, _ = target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert 'extern "C" void moist_dyn_gas_kokkos(' in cpp
    assert "Kokkos::RangePolicy<>(0, ncells)" in cpp
    assert "map_wtheta((df - 1), cell)" in cpp
    assert "const int *map_wtheta_data" in cpp
    assert "map_wtheta(map_wtheta_data, ndf_wtheta, ncells)" in cpp
    assert "const double recip_epsilon" in cpp

    assert "subroutine moist_dyn_gas_kokkos(" in fortran
    assert "bind(C)" in fortran
    # The interface imports the kinds its own arguments declare and no
    # others, so widening the ABI leaves an int/double region untouched.
    assert "use iso_c_binding, only : c_int, c_double" in fortran
    assert "c_float" not in fortran
    # The widths the C++ body assumed, asserted where the two languages meet.
    # The double-precision region needs this as much as the single-precision
    # one does: it is only correct while r_def really is 8 bytes.
    assert "use constants_mod, only : i_def, r_def" in fortran
    assert ("storage_size(1_i_def) == &\n        storage_size(1_c_int))), "
            "parameter :: assert_kind_i_def = 0" in fortran)
    assert ("storage_size(1.0_r_def) == &\n        "
            "storage_size(1.0_c_double))), parameter :: assert_kind_r_def = 0"
            in fortran)
    assert "dimension(*), intent(inout) :: moist_dyn_gas" in fortran
    assert "call moist_dyn_gas_kokkos(" in fortran
    assert "map_wtheta, loop0_stop, recip_epsilon)" in fortran
    assert "call moist_dyn_proxy%set_dirty()" in fortran
    assert fortran.index("call moist_dyn_gas_kokkos(") < fortran.index(
        "call moist_dyn_proxy%set_dirty()")
    assert "call moist_dyn_gas_code(" not in fortran
    assert str(psy.gen) == fortran


def test_lfric_kokkos_trans_derives_a_second_kernel_contract(second_target):
    """Nothing about the region is specific to the first kernel captured."""
    psy, loop, _ = second_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    # The region is named after the kernel, and its r_tran scalar is on the
    # ABI as a double even though the kernel the transformation was written
    # against has no scalar at all and never mentions r_tran.
    assert 'extern "C" void scaled_copy_kokkos(' in cpp
    assert "const double scaling" in cpp

    # Two function spaces, so two dofmaps, each given the cell extent the
    # PSy layer would otherwise have sliced away.
    assert "map_w3(map_w3_data, ndf_w3, ncells)" in cpp
    assert "map_wtheta(map_wtheta_data, ndf_wtheta, ncells)" in cpp
    assert "field_out(((map_w3((df - 1), cell) + k) - 1)) = " in cpp
    assert "(scaling * field_in(((map_wtheta((df - 1), cell) + k) - 1)))" \
        in cpp

    assert "subroutine scaled_copy_kokkos(" in fortran
    assert "real(c_double), value :: scaling" in fortran
    assert "dimension(*), intent(inout) :: field_out" in fortran
    assert "call scaled_copy_kokkos(" in fortran
    # No module constant, so the cell count is the last actual argument.
    assert "map_wtheta, loop0_stop)" in fortran
    assert "call scaled_copy_code(" not in fortran


def test_lfric_kokkos_trans_drops_an_orphaned_counter(second_target):
    """Capturing an invoke's only cell loop undeclares its counter.

    LFRic compiles the PSy layer with -Werror=unused-variable, so a counter
    left declared by the loop that has just been replaced is a build failure
    rather than untidiness.

    """
    psy, loop, _ = second_target

    LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "call scaled_copy_kokkos(" in fortran
    assert ":: cell" not in fortran


def test_lfric_kokkos_trans_keeps_a_shared_counter(paired_target):
    """A sibling cell loop still counting keeps the declaration."""
    psy, loop, _ = paired_target

    LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    # A Loop holds its counter as an attribute rather than as a Reference, so
    # the surviving loop is only visible to a check that asks the loops.
    assert "call scaled_copy_kokkos(" in fortran
    assert "call scaled_copy_code(" in fortran
    assert "integer(kind=i_def) :: cell" in fortran


def test_untransformed_invoke_can_be_generated_repeatedly(target):
    """Preparing a temporary generation copy must not mark the source tree."""
    psy, _, _ = target
    first = str(psy.gen)
    assert str(psy.gen) == first


def test_lfric_kokkos_trans_rejects_wrong_node():
    """Only an LFRic loop is a valid capture boundary."""
    with pytest.raises(TransformationError, match="LFRicLoop"):
        LFRicKokkosTrans().validate(CodeBlock(
            [], structure=CodeBlock.Structure.STATEMENT))


def test_lfric_kokkos_trans_rejects_halo_depth(target):
    """The prototype may not launch over a halo or redundant region."""
    _, loop, _ = target
    loop._upper_bound_halo_depth = Literal(
        "1", ScalarType(ScalarType.Intrinsic.INTEGER,
                        ScalarType.Precision.UNDEFINED))
    with pytest.raises(TransformationError, match="halo depth"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_continuous_write(target):
    """A continuous-space write would require colouring or atomics."""
    _, loop, kernel = target
    kernel.arguments.args[0].function_space._orig_name = "w0"
    with pytest.raises(TransformationError, match="discontinuous space"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_permits_a_continuous_read(target):
    """Only the written spaces decide whether cells may run in parallel."""
    _, loop, kernel = target
    kernel.arguments.args[1].function_space._orig_name = "w0"
    LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_quadrature(target):
    """Quadrature arrays are outside the first region contract."""
    _, loop, kernel = target
    kernel._basis_required = True
    kernel._qr_rules["gh_quadrature_xyoz"] = object()
    with pytest.raises(TransformationError, match="quadrature"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_operator(target):
    """Operator arguments are outside the first region contract."""
    _, loop, kernel = target
    kernel.arguments.args[1]._argument_type = "gh_operator"
    with pytest.raises(TransformationError, match="gh_operator"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_incremented_field(target):
    """An incremented field needs colouring or atomics, not a bare launch."""
    _, loop, kernel = target
    kernel.arguments.args[0]._access = AccessType.INC
    with pytest.raises(TransformationError, match="colouring or atomics"):
        LFRicKokkosTrans().validate(loop)


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

    assert ("Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> smap_sizes(smap_sizes_data, 4, ncells);" in cpp)
    assert ("Kokkos::View<const int****, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> smap(smap_data, ndf_w3, max_length, 4, ncells);" in cpp)
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


def test_lfric_kokkos_trans_refuses_a_one_dimensional_stencil(
        stencil_1d_target):
    """A 1-D stencil hands the kernel its size as a scalar, not an array.

    That is the difference the shape list is drawn along, and the fixture is a
    real CROSS kernel rather than a patched CROSS2D one so that the difference
    is the parser's rather than the test's. 'apply''s per-cell rule appends
    the cell index to an array actual; a 1-D stencil's size arrives instead as
    'field_in_stencil_size(cell)' against a by-value dummy, which would need a
    per-cell scalar argument kind that does not exist. No executed GungHo loop
    asks for one, so the shape is refused by name rather than mishandled.
    """
    _, loop, _ = stencil_1d_target

    with pytest.raises(TransformationError, match="cross2d") as error:
        LFRicKokkosTrans().validate(loop)

    assert "'cross'" in str(error.value)
    assert "'field_in'" in str(error.value)


def test_lfric_kokkos_trans_rejects_stencil(target):
    """Stencil storage and halo requirements are not silently captured.

    The refusal is shape-specific rather than blanket from stage 5 on:
    'cross2d' is accepted, and every other shape -- 'xory1d' here, which has a
    direction argument on top of a 1-D size -- is named in the message that
    refuses it.
    """
    _, loop, kernel = target
    kernel.arguments.args[1].stencil = LFRicArgStencil(name="xory1d")
    with pytest.raises(TransformationError, match="stencil"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_multiple_kernels(target, monkeypatch):
    """The launch body must correspond to exactly one kernel schedule."""
    _, loop, kernel = target
    monkeypatch.setattr(loop, "kernels", lambda: [kernel, kernel])
    with pytest.raises(TransformationError, match="exactly one kernel"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_codeblock(target):
    """Opaque kernel statements cannot cross the backend boundary."""
    _, loop, kernel = target
    kernel.get_callees()[0].addchild(
        CodeBlock([], structure=CodeBlock.Structure.STATEMENT))
    with pytest.raises(TransformationError, match="CodeBlock"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_unsupported_kind(target):
    """The fixed C ABI must fail closed if a Fortran kind changes."""
    _, loop, kernel = target
    schedule = kernel.get_callees()[0]
    schedule.symbol_table.lookup("nlayers").datatype = (
        ScalarType.integer_type())
    with pytest.raises(TransformationError, match="argument kinds"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_types_kinds_by_width(target, monkeypatch):
    """A kind reaches the ABI on its width, not on being called r_def.

    Transport kernels are written in r_tran and dynamics kernels in r_def,
    which is a distinction about what the model may reconfigure rather than
    about what reaches C. Narrowing r_def here is the same question asked from
    the other side: the name is unchanged, and what the ABI carries follows the
    width rather than the name.
    """
    _, loop, _ = target
    api_config = Config.get().api_conf("lfric")
    narrowed = dict(api_config.precision_map)
    narrowed["r_def"] = 4
    monkeypatch.setattr(api_config, "_precision_map", narrowed)
    cpp = LFRicKokkosTrans().apply(loop)
    assert "float *moist_dyn_gas_data" in cpp
    assert "const float *mr_v_data" in cpp
    assert "const float recip_epsilon" in cpp


def test_lfric_kokkos_trans_fails_closed_on_an_unsupported_width(
        target, monkeypatch):
    """Widening the ABI to float did not make it accept every kind.

    r_quad is a 16-byte real and is in the LFRic precision map, so the map
    resolves it and the C type table then has no entry for it. Asked through
    the same route as the test above, so that the two read as one question
    with two answers.
    """
    _, loop, _ = target
    api_config = Config.get().api_conf("lfric")
    widened = dict(api_config.precision_map)
    widened["r_def"] = widened["r_quad"]
    monkeypatch.setattr(api_config, "_precision_map", widened)
    with pytest.raises(TransformationError, match="argument kinds") as err:
        LFRicKokkosTrans().validate(loop)
    assert "4-byte integer, 4-byte real and 8-byte real" in str(err.value)


def test_lfric_kokkos_trans_passes_a_logical_by_conversion(logical_target):
    """A logical scalar crosses the ABI as a converted value, not a width.

    Stage 2 refused this because LFRic's l_def is kind(.false.) and measures 4
    bytes where PSyclone's precision map records it as 1 -- issue #1941 -- so a
    logical(c_bool) dummy against a logical(l_def) actual would not have
    compiled. Conversion dissolves the question rather than answering it: the
    dummy is logical(c_bool), value, the call site wraps the actual in
    LOGICAL(..., c_bool), and the compiler converts whatever width l_def turns
    out to be. Nothing here reads the precision map for a logical, so a
    corrected #1941 would not change what is generated -- which is why no
    width assertion is emitted for it either.
    """
    psy, loop, _ = logical_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const bool masked" in cpp
    assert "logical(c_bool), value :: masked" in fortran
    assert "use iso_c_binding, only : c_bool" in fortran
    assert "LOGICAL(masked, kind=c_bool)" in fortran
    # The whole point: no width is asserted for a kind that does not have to
    # match. An assertion here would fail on the very build this admits.
    assert "assert_kind_l_def" not in fortran
    assert "storage_size(.true._l_def)" not in fortran
    # l_def is dropped from the assertion block's own use line too, not only
    # from the assertions it would have fed. LFRic builds with
    # -Werror=unused-dummy-argument and friends, so importing a kind and then
    # not naming it is not a harmless extra line.
    assert "use constants_mod, only : i_def, r_solver" in fortran


def test_lfric_kokkos_trans_refuses_a_logical_array(logical_array_target):
    """A logical array stays refused, because it would cross by reference.

    Conversion is per value, so it is the scalar case that it fixes. A
    View<bool*> laid over logical(l_def) storage reinterprets rather than
    converts: it reads 1 byte where the Fortran wrote 4, so three quarters of
    the elements it returns are bytes from the middle of their neighbours.
    That is the failure the scalar's conversion removes and that an array
    cannot have removed by the same means, so the array is refused.
    """
    _, loop, _ = logical_array_target

    with pytest.raises(TransformationError, match="argument kinds") as error:
        LFRicKokkosTrans().validate(loop)

    assert "masked" in str(error.value)


def test_lfric_kokkos_trans_carries_single_precision_to_c(solver_target):
    """An r_solver kernel reaches C as float on both sides of the ABI.

    The fields are r_solver and the scalar is r_single -- two different 4-byte
    kinds -- so a widening that hardcoded one kind name would not generate
    this. What matters is that neither side promotes: the C++ says float and
    the bind(C) interface says real(c_float), so a single-precision build is
    honoured rather than silently widened to double.
    """
    psy, loop, _ = solver_target
    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "float *field_out_data" in cpp
    assert "const float *field_in_data" in cpp
    assert "const float scaling" in cpp
    assert "Kokkos::View<float*, Kokkos::LayoutLeft" in cpp
    assert "Kokkos::View<const float*, Kokkos::LayoutLeft" in cpp

    # Nothing but these two lines decides what precision the region computes
    # in: the local and the literal cross no interface, so the compiler
    # cannot check them and a promotion here would be silent.
    assert "float scaled;" in cpp
    assert "2.0f" in cpp

    assert "use iso_c_binding, only : c_int, c_float" in fortran
    assert ("real(c_float), dimension(*), intent(inout) :: field_out"
            in fortran)
    assert "real(c_float), dimension(*), intent(in) :: field_in" in fortran
    assert "real(c_float), value :: scaling" in fortran
    assert "c_double" not in fortran

    # The compiler checks the arguments, because the interface names a C kind
    # and the PSy layer names an LFRic one. It cannot check what the body
    # assumed, so the interface says so itself.
    assert "use constants_mod, only : i_def, r_single, r_solver" in fortran
    assert ("storage_size(1.0_r_solver) == &\n        "
            "storage_size(1.0_c_float))), parameter :: assert_kind_r_solver "
            "= 0" in fortran)
    assert ("storage_size(1.0_r_single) == &\n        "
            "storage_size(1.0_c_float))), parameter :: assert_kind_r_single "
            "= 0" in fortran)


def test_lfric_kokkos_trans_selects_the_matching_schedule(polymorphic_target):
    """The member the algorithm's precisions pick is the one captured.

    Before stage 3 this refused with "LFRicKokkosTrans requires exactly one
    kernel schedule."

    The interface declares its r_double member first, so returning
    ``schedules[0]`` would pass every test that only checked a region was
    generated. The assertions below are therefore about the *width*: the
    algorithm passes r_solver fields, which ``precision_map`` gives 4 bytes,
    so the match is r_single and the region is float throughout.
    """
    _, loop, kernel = polymorphic_target
    assert len(kernel.get_callees()) == 2
    schedule = LFRicKokkosTrans._schedule(kernel)
    assert schedule.name == "tri_scale_code_r_single"
    assert schedule is not kernel.get_callees()[0]

    cpp = LFRicKokkosTrans().apply(loop)
    assert "double" not in cpp
    assert "Kokkos::View<float*" in cpp
    assert "const float scaling" in cpp


def test_lfric_kokkos_trans_names_the_region_after_the_schedule(
        polymorphic_target):
    """One interface's members generate two differently named regions.

    What this protects is a build in which one interface is invoked at two
    precisions: naming both regions after the kernel would emit two launch
    symbols called ``tri_scale_kokkos`` whose arguments are float in one
    translation unit and double in the other.
    """
    _, loop, _ = polymorphic_target
    cpp = LFRicKokkosTrans().apply(loop)
    assert "tri_scale_r_single_kokkos" in cpp
    assert "tri_scale_kokkos" not in cpp


def test_lfric_kokkos_trans_refuses_when_no_schedule_matches(unmatched_target):
    """An interface no member of which the algorithm could have called.

    r_single is 4 bytes and r_quad is 16, against a ``field_type`` algorithm
    argument at r_def's 8. Nothing is monkeypatched: the widths are the ones
    ``psyclone.cfg`` records, so this is a case the matcher really rejects
    rather than one arranged to look rejected.
    """
    _, loop, _ = unmatched_target
    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "found no implementation of 'quad_scale_code'" in str(error.value)
    assert "out of 2" in str(error.value)


def test_lfric_kokkos_trans_refuses_an_ambiguous_interface(ambiguous_target):
    """An interface every member of which the algorithm could have called.

    ``precision_map`` gives r_single and r_solver the same 4 bytes, so the
    matcher cannot separate them; Fortran can, because it resolves by name.
    Refusing is deliberate. Picking either would be right only by coincidence,
    and the survey in psy-ir-aidev measures whether any GungHo kernel reaches
    this at all.
    """
    _, loop, _ = ambiguous_target
    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "found 2 implementations of 'dual_scale_code'" in str(error.value)
    assert "dual_scale_code_r_single, dual_scale_code_r_solver" in str(
        error.value)
    assert "will not choose between them" in str(error.value)


def test_lfric_kokkos_trans_refuses_when_the_matcher_cannot_answer(
        unmodelled_target):
    """A matcher that cannot be asked has not answered "no".

    ``KernelInterface`` builds the interface the metadata implies before it
    compares any precision, and PSyclone's issue #928 leaves stencils,
    evaluator shapes, CMA and inter-grid kernels among the parts it does not
    build, raising ``NotImplementedError``. Letting that be read as a mismatch
    would report a kind disagreement about a kernel whose kinds were never
    looked at; letting it escape would make a private helper of a
    transformation raise something other than ``TransformationError``.

    This calls ``_schedule`` rather than ``validate`` on purpose.
    ``_validate_kernel_metadata`` refuses a stencil first, so through
    ``validate`` this case is unreachable today -- which is exactly why the
    guard cannot rely on it. The survey in psy-ir-aidev calls ``_schedule``
    with no metadata check in front of it, because it reports every blocker of
    a loop rather than stopping at the first, and it is what found this.
    """
    _, _, kernel = unmodelled_target
    with pytest.raises(TransformationError) as error:
        # pylint: disable-next=protected-access
        LFRicKokkosTrans._schedule(kernel)
    assert ("cannot tell which of the 2 implementations of "
            "'stencil_scale_code'" in str(error.value))
    assert "outside what PSyclone's own matcher models" in str(error.value)
    assert "TODO #928" in str(error.value)


def test_lfric_kokkos_trans_asserts_nothing_about_an_unkinded_region():
    """A region naming no resolvable kind grows no use and no assertion.

    Every LFRic kernel names kinds, so ``apply`` cannot reach this; it is the
    guard that stops an empty table generating ``use constants_mod, only :``
    with nothing after it, which is a syntax error rather than a harmless
    no-op. ``_interface`` reads only the fields set here, so the schedule the
    region would otherwise carry is left out.
    """
    region = KokkosRegion(
        name="unkinded_kokkos", schedule=None, cell_count="ncells",
        arguments=(KokkosScalar("ncells", "int"),))
    interface = LFRicKokkosTrans._interface(region)

    assert "use iso_c_binding, only : c_int" in interface
    assert "constants_mod" not in interface
    assert "assert_kind" not in interface


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


def test_lfric_kokkos_trans_describes_each_local_array(local_target):
    """``_local_arrays`` names, types and sizes every automatic array."""
    _, _, kernel = local_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    scratch = LFRicKokkosTrans._local_arrays(schedule)

    assert [item.name for item in scratch] == ["partial", "swept"]
    assert all(item.c_type == "double" for item in scratch)
    assert all(item.extents == ("nlayers",) for item in scratch)
    # Fortran declares from 1 and C indexes from 0, as for a formal.
    assert all(item.index_offsets == (1,) for item in scratch)


def test_lfric_kokkos_trans_refuses_a_local_named_after_the_launch(
        team_name_target):
    """A local shadowing a name the team launch declares is refused.

    The launch index is renamed around such a collision instead, because a
    lambda parameter and a body declaration are a compile error and the fix
    is one name in two places. These seven are threaded through two launch
    shapes and through the scratch sizing, so they are refused rather than
    renamed; no GungHo kernel declares any of them.
    """
    _, loop, _ = team_name_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "generated launch declares 'team_size'" in str(error.value)


def test_lfric_kokkos_trans_refuses_a_local_named_ncells(
        ncells_local_target):
    """A local named for the cell count is refused whatever the launch shape.

    ``_validate_formals`` already refuses a *formal* of this name. The cell
    count is declared by the range launch as well as the team one, so this
    kernel has no local arrays: the refusal must not be conditional on there
    being scratch to place.
    """
    _, loop, _ = ncells_local_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "generated launch declares 'ncells'" in str(error.value)


def test_lfric_kokkos_trans_refuses_an_unsizable_local(unsized_local_target):
    """A local sized by a module constant is refused, with the extent named.

    ``_constants`` could import ``n_moist`` and pass it into the region, so
    the refusal is a deliberate narrowing rather than an inability: the
    scratch size is computed by the launch, outside the region that constant
    would be passed to.
    """
    _, loop, _ = unsized_local_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "n_moist" in str(error.value)
    assert "kernel-local array 'swept'" in str(error.value)
    assert "kernel argument" in str(error.value)


def test_lfric_kokkos_trans_refuses_a_local_of_an_unmapped_kind(
        unmapped_local_target):
    """A local array of a logical kind is refused, with the kind named."""
    _, loop, _ = unmapped_local_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "kernel-local array kinds" in str(error.value)
    assert "'rising'" in str(error.value)
    assert "'l_def'" in str(error.value)


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


def test_lfric_kokkos_trans_takes_a_literal_extent(literal_local_target):
    """A local declared ``dimension(nlayers,4)`` is captured, not refused.

    This is ``apply_helmholtz_operator_code``'s ``coeff`` in miniature, and
    the reason this widening exists: a literal bound refuses 179 of the 183
    loops the catalogue's ``local-array`` row counts, and that kernel is one
    of them.
    """
    psy, loop, kernel = literal_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._extents(
        schedule.symbol_table.lookup("swept")) == ("nlayers", "4")
    # The literal is not a kernel argument and must not be asked to be one.
    assert LFRicKokkosTrans._extent_names(
        schedule.symbol_table.lookup("swept")) == {"nlayers"}

    cpp = LFRicKokkosTrans().apply(loop)

    assert "swept_scratch_t::shmem_size(nlayers, 4)" in cpp
    assert "swept_scratch_t swept(team.team_scratch(0), nlayers, 4);" in cpp
    assert ("using swept_scratch_t = Kokkos::View<double**, "
            "Kokkos::LayoutLeft, ScratchSpace, Unmanaged>;" in cpp)
    # Both indices lose their Fortran base, not just the first.
    assert "swept((k - 1), (1 - 1))" in cpp
    # Nothing about it reaches the ABI, as for any other scratch array.
    assert "swept" not in str(psy.gen)


def test_lfric_kokkos_trans_takes_an_arithmetic_extent(
        arithmetic_local_target):
    """A local declared ``dimension(nlayers+1)`` is captured."""
    _, loop, kernel = arithmetic_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("swept")

    assert LFRicKokkosTrans._extents(symbol) == ("(nlayers + 1)",)
    assert LFRicKokkosTrans._extent_names(symbol) == {"nlayers"}

    cpp = LFRicKokkosTrans().apply(loop)

    assert "swept_scratch_t::shmem_size((nlayers + 1))" in cpp
    assert ("swept_scratch_t swept(team.team_scratch(0), (nlayers + 1));"
            in cpp)


def test_lfric_kokkos_trans_takes_an_explicit_lower_bound_of_one(
        explicit_one_local_target):
    """``dimension(1:nlayers)`` is accepted; the rule is about the value.

    A rule written against the source text would refuse this, since it is
    spelt like the ``dimension(0:nlayers-1)`` case that is out of reach. The
    lower bound is rendered and compared instead.
    """
    _, loop, kernel = explicit_one_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._extents(
        schedule.symbol_table.lookup("swept")) == ("nlayers",)

    cpp = LFRicKokkosTrans().apply(loop)

    assert "swept_scratch_t swept(team.team_scratch(0), nlayers);" in cpp


def test_lfric_kokkos_trans_refuses_a_lower_bound_that_is_not_one(
        lower_bound_local_target):
    """``dimension(0:nlayers-1)`` is refused, naming the lower bound.

    The generated View subtracts a fixed 1 from each Fortran index, so a
    different base would need index offsets to become expressions. That is a
    capability of its own; this is the boundary it starts at, asserted rather
    than assumed.
    """
    _, loop, _ = lower_bound_local_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'swept'" in str(error.value)
    assert "lower bound of 1" in str(error.value)
    assert "found '0'" in str(error.value)


def test_lfric_kokkos_trans_refuses_an_unsizable_expression(
        unsized_expression_target):
    """``dimension(n_moist+1)`` is refused, naming the module constant.

    Accepting arithmetic widened what an extent may be shaped like, not what
    may appear in one: the scratch size is still computed by the launch,
    where only kernel arguments are in scope.
    """
    _, loop, _ = unsized_expression_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "n_moist" in str(error.value)
    assert "kernel-local array 'swept'" in str(error.value)
    assert "kernel argument" in str(error.value)


@pytest.mark.parametrize("fixture_name", [
    "literal_local_target", "arithmetic_local_target",
    "explicit_one_local_target"])
def test_lfric_kokkos_trans_validate_accepts_what_apply_generates(
        fixture_name, request):
    """Every widened shape passes ``validate`` as well as ``apply``.

    The agreement between the two is the property Task 3a.4 established, and
    it holds in the accepting direction as well as the refusing one: a
    transformation whose ``validate`` were stricter than its ``apply`` would
    report a capturable loop as blocked and understate its own coverage.
    """
    _, loop, _ = request.getfixturevalue(fixture_name)
    trans = LFRicKokkosTrans()

    trans.validate(loop)
    assert trans.apply(loop)


@pytest.mark.parametrize("fixture_name", [
    "unsized_local_target", "unmapped_local_target",
    "lower_bound_local_target", "unsized_expression_target"])
def test_lfric_kokkos_trans_validate_and_apply_agree_on_locals(
        fixture_name, request):
    """Both refusals are made by ``validate``, not discovered by ``apply``.

    This is the property that was false before local arrays were modelled:
    ``validate`` accepted a kernel like ``tri_solve`` and ``apply`` then
    raised from the backend. A caller asking whether a loop is capturable got
    "yes" and a caller capturing it got an error, about the same loop.
    """
    _, loop, _ = request.getfixturevalue(fixture_name)
    trans = LFRicKokkosTrans()

    with pytest.raises(TransformationError) as predicted:
        trans.validate(loop)
    with pytest.raises(TransformationError) as attempted:
        trans.apply(loop)

    assert str(predicted.value) == str(attempted.value)


def test_lfric_kokkos_trans_resolves_bounds_from_the_declaration(
        bound_target):
    """LBOUND, UBOUND and SIZE become the bounds the kernel declared.

    Each is answered symbolically, against the symbol table, so the generated
    region carries the declared bound as an expression rather than a call.
    The loop is the evidence: its start is ``LBOUND(partial, 1) + 1`` and its
    end is ``UBOUND(partial, 1)``, and both have to have gone before the
    backend sees them.
    """
    _, loop, _ = bound_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "for(k=(1 + 1); k<=nlayers; k+=1)" in cpp
    # SIZE of a formal declared dimension(undf_w3), read in an expression
    # rather than as a bound.
    assert "(swept((k - 1)) + undf_w3)" in cpp
    # Nothing of the enquiries survives into the region. A bare "size(" would
    # match shmem_size, team_size and league_size, so the one call the kernel
    # made is named instead.
    for name in ("LBOUND", "UBOUND", "SIZE", "lbound(", "ubound(",
                 "size(field_in)"):
        assert name not in cpp


def test_lfric_kokkos_trans_resolves_a_bound_in_a_later_dimension(
        rank_two_bound_target):
    """``SIZE(swept, 2)`` takes the second entry of the declared shape.

    A rank-2 local declared ``dimension(nlayers,4)`` answers its second
    dimension with a literal, so the substitution has to index the shape
    rather than assume the first entry.
    """
    _, loop, _ = rank_two_bound_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "(partial((k - 1)) * 4)" in cpp
    assert "swept_scratch_t::shmem_size(nlayers, 4)" in cpp


def test_lfric_kokkos_trans_resolves_a_lowered_full_section(
        full_section_target):
    """The bounds the lowering itself writes are substituted too.

    ``difference(:)`` is rewritten to an explicit loop by
    ``ArrayAssignment2LoopsTrans``, which writes ``LBOUND`` and ``UBOUND``
    into that loop's bounds. They are produced by the transformation rather
    than by the kernel author, which is why the substitution runs after the
    lowering and not before it.
    """
    _, loop, _ = full_section_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert ("Kokkos::parallel_for(Kokkos::TeamVectorRange"
            "(team, 1, undf_w3 + 1)," in cpp)
    assert "difference((idx - 1)) = (difference((idx - 1)) + (b_idx + nl));" \
        in cpp


def test_lfric_kokkos_trans_refuses_a_bound_of_a_scalar(scalar_bound_target):
    """``SIZE`` of a scalar is refused, naming the symbol.

    fparser2 parses it, the check that would refuse it being semantic rather
    than syntactic, so the transformation is the first thing in the chain
    that can notice.
    """
    _, loop, _ = scalar_bound_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot resolve 'SIZE' of 'nlayers', which is not declared as an "
            "array" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_bound_of_an_element(
        element_bound_target):
    """``UBOUND(map_w3(1), 1)`` asks about an element, not about the array.

    An ``ArrayReference`` is a ``Reference``, so a test that accepted any
    reference would read this as the array's bound and generate the wrong
    answer silently.
    """
    _, loop, _ = element_bound_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires the first argument of 'UBOUND' to be a plain reference "
            "to a declared array" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_variable_dimension(
        variable_bound_target):
    """A dimension given by a variable cannot be resolved from the table."""
    _, loop, _ = variable_bound_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires the dimension of 'UBOUND' of 'partial' to be an "
            "integer literal" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_dimension_past_the_rank(
        out_of_range_bound_target):
    """A dimension outside the declared rank is refused rather than indexed."""
    _, loop, _ = out_of_range_bound_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot resolve 'UBOUND' of 'partial' in dimension 2, since it is "
            "declared with rank 1" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_whole_size_above_rank_one(
        whole_size_target):
    """``SIZE(a)`` on a rank-2 array is a count, not a bound.

    It is valid Fortran and has a well-defined value, so the refusal is about
    what the region can express rather than about the source being wrong: no
    one entry of the declared shape answers it.
    """
    _, loop, _ = whole_size_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires 'SIZE' of 'swept' to name a dimension, since 'swept' is "
            "declared with rank 2" in str(error.value))


def test_lfric_kokkos_trans_inherits_the_extent_grammar_refusal(
        lower_bound_enquiry_target):
    """A bound of an array based other than at 1 gets the grammar's message.

    ``_extents`` already refuses that declaration, and the enquiry depends on
    the same bounds, so its refusal is passed through rather than paraphrased
    -- a reader gets the sentence that says which rule was broken.
    """
    _, loop, _ = lower_bound_enquiry_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires 'swept' to be declared with a lower bound of 1, but "
            "found '0'" in str(error.value))


@pytest.mark.parametrize("fixture_name", [
    "scalar_bound_target", "element_bound_target", "variable_bound_target",
    "out_of_range_bound_target", "whole_size_target"])
def test_lfric_kokkos_trans_validate_and_apply_agree_on_bounds(
        fixture_name, request):
    """Every bound refusal is made by ``validate``, not found by ``apply``."""
    _, loop, _ = request.getfixturevalue(fixture_name)
    trans = LFRicKokkosTrans()

    with pytest.raises(TransformationError) as predicted:
        trans.validate(loop)
    with pytest.raises(TransformationError) as attempted:
        trans.apply(loop)

    assert str(predicted.value) == str(attempted.value)


def test_lfric_kokkos_trans_validate_leaves_bounds_alone(bound_target):
    """``validate`` predicts the substitution over a copy.

    The schedule it is handed is cached by ``LFRicKern.get_callees``, so a
    substitution made during validation would persist and a second call would
    see a schedule with no enquiries left in it.
    """
    _, loop, kernel = bound_target

    LFRicKokkosTrans().validate(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert any(call.intrinsic in LFRicKokkosTrans._BOUND_INTRINSICS
               for call in schedule.walk(IntrinsicCall))


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_bounds_defer_to_the_section_refusal(
        tmp_path, clear_module_manager_instance):
    """An unlowerable section is not re-reported as a bound problem.

    ``_validate_bounds`` has to lower a copy before it can see the bounds the
    lowering writes, so a schedule the lowering refuses leaves it with nothing
    to check. ``_validate_sections`` owns that refusal, and the coverage
    survey calls each predicate independently, so reporting it twice would
    make one fact look like two blocked patterns.
    """
    # pylint: disable-next=unused-variable
    _, loop, kernel = _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM,
        _DEPENDENT_SECTION_KERNEL)
    schedule = LFRicKokkosTrans._schedule(kernel)

    LFRicKokkosTrans._validate_bounds(schedule)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_sections(schedule)
    assert "cannot lower an array section to a loop" in str(error.value)


def test_lfric_kokkos_trans_writes_a_declared_constant_as_its_value(
        static_constant_target):
    """A module-level parameter reaches the region as its literal.

    It has to: the kernel module is ``private`` and publishes only its
    metadata and its ``_code`` routine, so ``use column_solve_kernel_mod,
    only: nfaces`` in the PSy layer would not compile. The value is stated in
    the declaration, so the region carries the value.
    """
    _, loop, _ = static_constant_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "partial((k - 1)) * 4" in cpp
    assert "+ 1.0e-9" in cpp
    assert "nfaces" not in cpp
    assert "tol" not in cpp


def test_lfric_kokkos_trans_does_not_import_a_declared_constant(
        static_constant_target):
    """The PSy layer gains no import for a constant written in as a value."""
    psy, loop, _ = static_constant_target

    LFRicKokkosTrans().apply(loop)

    generated = str(psy.gen)
    assert "nfaces" not in generated
    assert "tol" not in generated


def test_lfric_kokkos_trans_refuses_a_declared_array_constant(
        array_constant_target):
    """An array parameter has no single literal to write in.

    It is not a scalar the ABI could carry either, so it is refused rather
    than silently reaching the region as one element of itself.
    """
    _, loop, _ = array_constant_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'x_dofs': a module-level constant is written "
            "into the region as its value, and this one was not declared "
            "with a literal value" in str(error.value))


def test_lfric_kokkos_trans_casts_at_the_kind_the_body_names(cast_kind_target):
    """A kind spelled in a cast is a type name, not data to pass by value.

    ``real(k, r_def)`` puts ``r_def`` into the tree as a Reference like any
    other. The writer consumes it as the cast's target and never emits it, so
    there is nothing for the PSy layer to import or pass.
    """
    _, loop, _ = cast_kind_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "(double)k" in cpp
    assert "r_def" not in cpp


def test_lfric_kokkos_trans_casts_at_a_kind_no_declaration_repeats(
        undeclared_cast_kind_target):
    """A cast's own kind sets its width even when nothing else names it.

    ``r_second`` is 8 bytes and appears only as this cast's target. Reading
    the width from declarations alone would leave the backend with no entry
    for it, and the C writer's default would silently make it ``float``.
    """
    _, loop, _ = undeclared_cast_kind_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "(double)k" in cpp
    assert "(float)k" not in cpp


def test_lfric_kokkos_trans_passes_a_logical_constant_by_conversion(
        off_abi_constant_target):
    """An imported logical constant crosses by the same conversion.

    ``rehabilitate`` is an ``l_def`` logical in ``planet_config_mod``, read by
    the kernel in an ``if``. It reaches the region as an argument rather than a
    literal, because its value is only known where the PSy layer runs, so the
    same wrapping applies to it as to a formal -- which is the point of doing
    the wrapping over ``region.arguments`` rather than over the formals alone.
    """
    psy, loop, _ = off_abi_constant_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const bool rehabilitate" in cpp
    assert "if (rehabilitate)" in cpp
    assert "use planet_config_mod, only : rehabilitate" in fortran
    assert "LOGICAL(rehabilitate, kind=c_bool)" in fortran


def test_lfric_kokkos_trans_refuses_a_constant_of_a_kind_off_the_abi(
        unmapped_constant_target):
    """A constant that resolves can still have no place on the interface.

    This case used to be carried by ``rehabilitate``, an ``l_def`` logical,
    which stage 5 admits. The refusal itself did not change, so it keeps a
    witness of a kind that is still off the ABI: ``unmapped_width`` is an
    ``r_quad`` real, 16 bytes where the ABI carries 4 and 8. The
    message names the kinds the ABI does carry, which now includes the
    logical clause.
    """
    _, loop, _ = unmapped_constant_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot pass 'unmapped_width' from 'planet_config_mod' by value"
            in str(error.value))
    assert "4-byte integer" in str(error.value)
    assert "and logical of any kind" in str(error.value)


def test_lfric_kokkos_trans_names_the_module_it_could_not_read(
        unreadable_constant_target):
    """An unreadable container is named rather than guessed around.

    The kind of an imported constant is stated only in its own module, so a
    module PSyclone cannot read leaves no width for the ABI. Choosing one
    would put a silently wrong type on the interface, so the refusal says
    which module is missing and leaves the search path to the caller.
    """
    _, loop, _ = unreadable_constant_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot type 'eps' without the source of "
            "'unreadable_constants_mod'" in str(error.value))
    assert "module search path" in str(error.value)


def test_lfric_kokkos_trans_leaves_out_a_kind_the_map_does_not_carry(
        unmapped_cast_kind_target):
    """A cast kind with no width recorded is left out, not written in as none.

    ``r_native`` is the compiler's own default rather than one of the widths
    the LFRic precision map names, so there is nothing to record. Storing the
    failed lookup would be worse than leaving it out: the table is keyed by
    kind name, and a declaration may already have resolved the same name.
    """
    _, loop, kernel = unmapped_cast_kind_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    kinds = dict(LFRicKokkosTrans._kind_types(schedule))

    assert "r_native" not in kinds
    assert kinds["r_def"] == "double"
    assert "r_native" not in LFRicKokkosTrans().apply(loop)


def test_lfric_kokkos_trans_names_a_called_routine_as_a_call(
        called_routine_target):
    """A function in an unread module is refused as a call, once.

    Without the module the frontend cannot tell ``helper(k)`` from an array
    reference, so the symbol is a plain ``Symbol`` and the constant machinery
    used to claim it as module data it could not pass by value. The survey
    calls each predicate independently, so that made one fact look like two
    blocked patterns.
    """
    _, loop, kernel = called_routine_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    LFRicKokkosTrans._constants(schedule)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert ("cannot capture the call to 'helper'" in str(error.value))


# pylint: disable-next=unused-argument
def test_lfric_kokkos_trans_refuses_a_resolved_routine_as_a_constant(
        tmp_path, clear_module_manager_instance):
    """A symbol that resolves to a routine is refused as one.

    ``_constants`` skips a call's own routine reference, but a name used
    another way -- as a procedure argument, say -- reaches
    ``_describe_constant``, and only becomes known to be a routine when
    ``resolve_type`` specialises it. Reading the module is what makes the
    difference, so this test provides one.
    """
    (tmp_path / "helper_mod.f90").write_text(_HELPER_MODULE)
    Config.get().include_paths = [str(tmp_path)]
    ModuleManager.get().add_search_path(str(tmp_path))
    symbol = Symbol(
        "helper", interface=ImportInterface(ContainerSymbol("helper_mod")))

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._describe_constant(symbol)

    assert ("cannot capture 'helper' from 'helper_mod': it is a routine "
            "rather than data" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_declared_module_variable(
        static_variable_target):
    """A module variable is not a constant however it was initialised.

    ``real(kind=r_def) :: cached_tol = 1.0e-9_r_def`` carries a literal in
    its declaration exactly as the ``parameter`` beside it does, and the
    module may assign to it afterwards. Writing the initialisation into the
    region would freeze whatever value the module happened to start with.
    """
    _, loop, _ = static_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'cached_tol': it is neither a kernel argument "
            "nor imported from a module" in str(error.value))


def test_lfric_kokkos_trans_passes_a_repeated_import_once(
        repeated_import_target):
    """An imported constant read twice reaches the ABI once.

    Each reference is a separate node, so the walk meets ``recip_epsilon``
    twice; describing it twice would re-resolve a symbol already on the ABI
    and hand the PSy layer the same argument in two places.
    """
    psy, loop, kernel = repeated_import_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._constants(schedule) == [
        ("recip_epsilon", "planet_config_mod", "double")]

    LFRicKokkosTrans().apply(loop)

    generated = str(psy.gen)
    assert generated.count("map_wtheta, loop0_stop, recip_epsilon)") == 1


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


def test_team_size_option_reaches_the_launch(level_target):
    """``team_size`` replaces ``Kokkos::AUTO`` in the policy.

    On the OpenMP backend ``Kokkos::AUTO`` is one member, so a host build
    reaches the team-level concurrency only by asking for a size. The option
    is the only way to ask.
    """
    _, loop, _ = level_target

    cpp = LFRicKokkosTrans().apply(loop, options={"team_size": 4})

    assert "TeamPolicy(ncells, 4)," in cpp
    assert "Kokkos::AUTO" not in cpp


@pytest.mark.parametrize("value", ["4", 4.0, True, 0, -1])
def test_team_size_option_is_validated(level_target, value):
    """A team size that is not a positive integer is refused.

    ``True`` is in the list because ``isinstance(True, int)`` is true in
    Python, so a bare integer check would let it through and write
    ``TeamPolicy(ncells, True)``.
    """
    _, loop, _ = level_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop, options={"team_size": value})

    assert ("'team_size' option must be a positive integer, but found "
            f"'{value}'" in str(error.value))


def test_team_is_reserved_for_a_hierarchical_kernel_without_scratch(
        team_level_target):
    """``team`` is refused for a kernel the loop selection alone reaches.

    Until this stage the seven generated names were reserved only for a
    kernel with a local array, because only scratch reached a launch that
    declares ``team``. The hierarchical launch declares it whether or not
    there is scratch, so the reservation follows the launch rather than the
    scratch.
    """
    _, loop, _ = team_level_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("generated launch declares 'team', but the kernel declares a "
            "local of that name" in str(error.value))
