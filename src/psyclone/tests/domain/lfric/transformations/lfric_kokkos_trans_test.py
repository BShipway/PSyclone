# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for the deliberately narrow LFRic-to-Kokkos transformation."""

# pylint: disable=protected-access

import re

import pytest

from psyclone.configuration import Config
from psyclone.core import AccessType
from psyclone.domain.lfric import KernCallArgList, LFRicLoop
from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.errors import GenerationError
from psyclone.lfric import LFRicArgStencil
from psyclone.parse import ModuleManager
from psyclone.parse.algorithm import parse
from psyclone.psyGen import PSyFactory
from psyclone.psyir.backend.kokkos import KokkosRegion, KokkosScalar
from psyclone.psyir.nodes import (
    ArrayConstructor, ArrayReference, Call, CodeBlock, Exit, IntrinsicCall,
    Literal, Range, Reference)
from psyclone.psyir.symbols import (
    ArrayType, ContainerSymbol, DataSymbol, ImportInterface, ScalarType,
    StaticInterface, Symbol, SymbolError, UnsupportedFortranType)
from psyclone.psyir.transformations import TransformationError
from psyclone.transformations import LFRicColourTrans


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


# The logical fixture's kernel with the kind name taken off its formal. LFRic
# defines l_def as kind(.false.), which is the default logical kind, so a
# kernel declaring 'logical(kind=l_def)' and one declaring a plain 'logical'
# declare the same thing; GungHo writes both, and calc_dz_face_code writes the
# second. The missing name costs nothing here because a logical crosses the
# ABI by conversion: there is no width to look up, so none to fail to find.
_DEFAULT_LOGICAL_KERNEL = _LOGICAL_KERNEL.replace(
    "  use constants_mod, only : i_def, l_def, r_solver",
    "  use constants_mod, only : i_def, r_solver").replace(
    "    logical(kind=l_def), intent(in) :: masked",
    "    logical, intent(in) :: masked")


# average_w3_to_w0_code's shape: every one of the seven integers LFRic's
# argument ordering supplies -- the layer count, and a dof count, a dof total
# and a dofmap for each of the two function spaces -- declared with no kind at
# all. That is not a contrivance; the kernel is written that way in
# lfric_apps, and it is the whole of what stands between it and the ABI.
#
# Unlike a logical, an integer does cross the ABI as a width, and the default
# integer's width is stated in no kind parameter the precision map could be
# asked about. It is asserted instead, against the compiler that will build
# the generated code: see the assertion this fixture's test looks for.
_DEFAULT_INTEGER_ALGORITHM = """
program kokkos_default_integer_test
  use field_mod, only : field_type
  use average_w3_to_w0_kernel_mod, only : average_w3_to_w0_kernel_type
  implicit none
  type(field_type) :: coarse_field, fine_field
  call invoke(average_w3_to_w0_kernel_type(coarse_field, fine_field))
end program kokkos_default_integer_test
"""


_DEFAULT_INTEGER_KERNEL = """
module average_w3_to_w0_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column
  use constants_mod, only : r_def
  use fs_continuity_mod, only : w0, w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: average_w3_to_w0_kernel_type
    type(arg_type) :: meta_args(2) = (/                              &
         arg_type(gh_field, gh_real, gh_write, w3),                  &
         arg_type(gh_field, gh_real, gh_read,  w0) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: average_w3_to_w0_code
  end type average_w3_to_w0_kernel_type
contains
  subroutine average_w3_to_w0_code(nlayers, field_w3, field_w0, &
                                   ndf_w3, undf_w3, map_w3,     &
                                   ndf_w0, undf_w0, map_w0)
    integer, intent(in) :: nlayers, ndf_w3, undf_w3, ndf_w0, undf_w0
    integer, dimension(ndf_w3), intent(in) :: map_w3
    integer, dimension(ndf_w0), intent(in) :: map_w0
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_w3
    real(kind=r_def), dimension(undf_w0), intent(in) :: field_w0
    integer :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w0
        field_w3(map_w3(1) + k) = field_w3(map_w3(1) + k) + &
                                  field_w0(map_w0(df) + k)
      end do
    end do
  end subroutine average_w3_to_w0_code
end module average_w3_to_w0_kernel_mod
"""


# The single-precision fixture's kernel with the kind name taken off its real.
# A real is refused where an integer and a logical are admitted, and the
# asymmetry is deliberate: LFRic names a kind on every real it means -- r_def,
# r_solver, r_single and r_tran are all in use and all different -- so a real
# declared with no kind is more likely an oversight than a default, and the
# default a compiler picks for it is the one width nobody wrote down.
_DEFAULT_REAL_KERNEL = _SOLVER_KERNEL.replace(
    "  use constants_mod, only : i_def, r_single, r_solver",
    "  use constants_mod, only : i_def, r_solver").replace(
    "    real(kind=r_single), intent(in) :: scaling",
    "    real, intent(in) :: scaling")


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


_IMPLICIT_ALGORITHM = """
program kokkos_implicit_test
  use field_mod, only : field_type
  use normals_sum_kernel_mod, only : normals_sum_kernel_type
  implicit none
  type(field_type) :: field_out, field_in
  call invoke(normals_sum_kernel_type(field_out, field_in))
end program kokkos_implicit_test
"""


# The boundary-condition shape: a reference-element property and a mesh
# property, each declared by the kernel with the extent left out. Every one of
# the six loops the catalogue counts under "implicit extent" is this --
# 'weighted_div_bd_code' writes 'real(kind=r_def), intent(in) ::
# outward_normals(:,:)' beside 'integer(kind=i_def), intent(in) ::
# adjacent_face(:)' -- and the extent the declaration leaves out is not
# unknown: the PSy layer holds the array it passes and can measure it.
#
# The two are here together because they are measured differently. The
# reference-element array is passed whole, so the region's View has the two
# extents the actual has; the mesh property is passed one cell's column at a
# time, so the region takes it whole and the extent measured is the one the
# slice leaves. 'nfaces_re_h' is declared and unused, as LFRic's own kernels
# declare it: the argument order is the metadata's rather than the body's.
_IMPLICIT_KERNEL = """
module normals_sum_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_write, gh_read, &
                           cell_column, reference_element_data_type,       &
                           mesh_data_type, adjacent_face,                  &
                           outward_normals_to_horizontal_faces
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: normals_sum_kernel_type
    type(arg_type) :: meta_args(2) = (/                                &
         arg_type(gh_field, gh_real, gh_write, w3),                    &
         arg_type(gh_field, gh_real, gh_read,  w3) /)
    type(reference_element_data_type) :: meta_reference_element(1) = (/ &
         reference_element_data_type(                                   &
             outward_normals_to_horizontal_faces) /)
    type(mesh_data_type) :: meta_mesh(1) = (/                           &
         mesh_data_type(adjacent_face) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: normals_sum_code
  end type normals_sum_kernel_type
contains
  subroutine normals_sum_code(nlayers, field_out, field_in,   &
                              ndf_w3, undf_w3, map_w3,        &
                              nfaces_re_h, outward_normals,   &
                              adjacent_face)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: nfaces_re_h
    real(kind=r_def), intent(in) :: outward_normals(:,:)
    integer(kind=i_def), intent(in) :: adjacent_face(:)
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def) :: k, df, face
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        field_out(map_w3(df) + k) = field_in(map_w3(df) + k)
        do face = 1, size(adjacent_face, 1)
          field_out(map_w3(df) + k) = field_out(map_w3(df) + k) + &
              outward_normals(1, face) *                         &
              outward_normals(2, adjacent_face(face))
        end do
      end do
    end do
  end subroutine normals_sum_code
end module normals_sum_kernel_mod
"""


# The same kernel asking for the bounds themselves rather than for the size.
# Fortran fixes the lower bound of an assumed-shape dummy at 1 whatever the
# actual was declared with, so 'lbound' is the literal 1 and 'ubound' is the
# measured extent -- the one place where the declared origin A3 reads is the
# dummy's own and not the actual's.
_IMPLICIT_BOUND_KERNEL = _IMPLICIT_KERNEL.replace(
    "do face = 1, size(adjacent_face, 1)",
    "do face = lbound(adjacent_face, 1), ubound(adjacent_face, 1)")


# An assumed shape whose declaration states its lower bound. Fortran allows
# it, and it is the one assumed shape that stays refused: the extent would be
# taken from the actual and the origin from the declaration, so the View's
# shape would be read out of two places at once.
_IMPLICIT_ORIGIN_KERNEL = _IMPLICIT_KERNEL.replace(
    "intent(in) :: adjacent_face(:)",
    "intent(in) :: adjacent_face(0:)")


# An assumed shape that is not a kernel argument, and so has no call to be
# measured through. The declaration is not legal Fortran outside a dummy
# argument list, which is the point: there is no route by which a local can
# acquire a shape from a caller, so the region has nothing to size a scratch
# View by and refuses rather than guessing one.
_IMPLICIT_LOCAL_KERNEL = _IMPLICIT_KERNEL.replace(
    "    integer(kind=i_def) :: k, df, face\n",
    "    integer(kind=i_def) :: k, df, face\n"
    "    real(kind=r_def), dimension(:) :: loose\n")


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


# The same kernel with a lower bound that is not 1, and an upper bound that
# is an expression. Both the extent and the origin are read from the one
# declaration, so a shape that moves the origin and computes the extent at
# once is the case where reading them apart would disagree.
_LOWER_BOUND_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(0:nlayers-1) :: swept")


# The shape the catalogue's `array-bound` row actually counts. `u_e`, `u_av`
# and `pert` are each declared `dimension(0:nlayers)` in the kernels that row
# blocks, so the View is one element longer than its upper bound and every
# subscript of it is shifted by nothing rather than by one.
_ZERO_BASED_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(0:nlayers) :: u_e").replace(
    "swept(", "u_e(")


# The same array asked for all three of its shape enquiries. Each is answered
# from the declaration, and each answer moves with the origin: LBOUND is no
# longer the constant 1 and SIZE is no longer the upper bound.
_ZERO_BASED_ENQUIRY_KERNEL = _ZERO_BASED_LOCAL_KERNEL.replace(
    "    do k = 2, nlayers",
    "    do k = lbound(u_e, 1) + 2, ubound(u_e, 1)").replace(
    "      field_out(map_w3(1) + k - 1) = u_e(k)",
    "      field_out(map_w3(1) + k - 1) = u_e(k) + size(u_e, 1)")


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


# A local whose shape its declaration does not carry. Every one of the fifteen
# rows the catalogue counts under "explicit bounds" is this: an allocatable
# whose size is stated by an ALLOCATE in the body, over values the kernel
# computes for itself. A scratch size is computed before the launch enters the
# region, so there is nowhere for such a size to come from.
_SHAPELESS_LOCAL_KERNEL = _LOCAL_KERNEL.replace(
    "    real(kind=r_def), dimension(nlayers) :: swept",
    "    real(kind=r_def), allocatable, dimension(:) :: swept").replace(
    "    swept(nlayers) = partial(nlayers)",
    "    allocate( swept(nlayers) )\n"
    "    swept(nlayers) = partial(nlayers)").replace(
    "  end subroutine column_solve_code",
    "    deallocate( swept )\n"
    "  end subroutine column_solve_code")


# An extent that is not an integer expression over named sizes. The origin is
# the Fortran default here, so this is the extent grammar's own refusal rather
# than the origin's, asserted apart from it because the two are checked in
# order and the first to fire hides the second. The exponent is a name rather
# than a literal because a literal one is not unrenderable any more: the C
# writer expands `nlayers**2` into `(nlayers * nlayers)`, which is an integer
# expression over named sizes and is accepted. `nlayers**nlayers` still has to
# be written `pow(nlayers, nlayers)`, and the comma is what a `shmem_size`
# argument may not carry.
_UNRENDERABLE_EXTENT_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(nlayers**nlayers) :: swept")


# The same declaration with a literal exponent, which is not unrenderable.
# `nlayers**2` was refused as long as the C writer wrote an integer power as
# `pow`; it is written as a product now, and a product over named sizes is
# exactly what a `shmem_size` argument may be. The pair is here so that the
# refusal above is read as being about the comma rather than about the power.
_SQUARED_EXTENT_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(nlayers**2) :: swept")


# An origin that is not an integer expression over named sizes. The grammar is
# the extent's, and it is applied to the origin for the reason the two are
# read together: an origin the region cannot write shifts every subscript of
# the array rather than sizing it wrongly.
_UNRENDERABLE_ORIGIN_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept",
    "dimension(nlayers**nlayers:nlayers) :: swept")


# A declared shape the C writer has no way to render at all. GungHo writes
# one: ffsl_flux_z_nirvana_kernel_mod declares
# `field_local_upper(MAX(nlayers-monotone_above,1), 3)`. The back-end's own
# failure is a VisitorError, which `validate` may not raise, so it is turned
# into a refusal that names the array.
_UNWRITABLE_SHAPE_KERNEL = _LOCAL_KERNEL.replace(
    "dimension(nlayers) :: swept", "dimension(max(nlayers,1)) :: swept")


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


# UBOUND of an array whose declared origin cannot be written. The bounds
# grammar already refuses that declaration, and the enquiry inherits the
# refusal rather than paraphrasing it, because the answer depends on the same
# bounds.
_LOWER_BOUND_ENQUIRY_KERNEL = _UNRENDERABLE_ORIGIN_KERNEL.replace(
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


# The same kernel passing a column of the local to a routine, which is how
# convert_hdiv_native_code reaches native_jacobian. The section is outside an
# assignment and so beyond the lowering, but the call is beyond the capture
# altogether, and that is the reason worth reporting.
_HDIV_CALL_KERNEL = _HDIV_SECTION_KERNEL.replace(
    "    difference(w3_idx : w3_idx + nl) = vector(:,1)",
    "    call native_jacobian(vector(:,1))\n"
    "    difference(w3_idx : w3_idx + nl) = vector(:,1)")


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


# The same, with the constant declared as an array. `face_order(4)` is
# gungho's shape: a parameter array of a kernel's own module, indexed by a
# value the loop computes, so there is no single element to substitute. The
# module is private, as a kernel module is, which is why the values go into
# the generated unit rather than across the ABI.
_ARRAY_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  integer(kind=i_def), parameter :: face_order(4) = [1, 2, 3, 4]\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(face_order(k)) "
    "* face_order(1)")


# An array parameter whose values are not there to be read: `reshape` is a
# call, so there is no element list to fold and nothing to declare.
_RESHAPED_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  integer(kind=i_def), parameter :: point(2, 2) = &\n"
    "      reshape((/ 1, 2, 3, 4 /), (/ 2, 2 /))\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(point(1, 1))")


# A scalar parameter declared as an arithmetic over other parameters, one of
# them from another module. None of the names is data the region could read,
# but the value they state is one it can carry.
_FOLDED_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : n_moist").replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  integer(kind=i_def), parameter :: n_extra = 2\n"
    "  real(kind=r_def), parameter :: weight = &\n"
    "      1.0_r_def / (n_moist + n_extra)\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - weight * partial(k)")


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


# A kernel importing a module constant declared with no kind. PROTECTED leaves
# the frontend with the declaration text rather than a typed symbol, so this
# reaches the ABI by the declaration reader rather than by the symbol's own
# type -- the second of the two routes an unkinded declaration can arrive by,
# and the one that would otherwise still refuse it.
_DEFAULT_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod, only : quenching").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      if (quenching) swept(k) = swept(k + 1) - partial(k)")


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


# A second module holding the sweep, and holding state of its own that the
# sweep reads. This is the shape coordinate_jacobian has -- a GungHo helper in
# a module the kernel `use`s, reading a rotation matrix that module keeps --
# and it is the shape inlining cannot reach: the body could be moved but the
# datum it reads could not, since the kernel's module neither declares it nor
# has any name for it.
_EXTERNAL_STATE_MODULE = """
module helper_state_mod
  use constants_mod, only : i_def, r_def
  implicit none
  private
  real(kind=r_def) :: relaxation = 0.5_r_def
  public :: sweep_column
contains
  subroutine sweep_column(n, source, result)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    integer(kind=i_def) :: j
    result(n) = source(n)
    do j = n - 1, 1, -1
      result(j) = result(j + 1) - relaxation * source(j)
    end do
  end subroutine sweep_column
end module helper_state_mod
"""


# The kernel that calls it. Written the same way as _MODULE_PROCEDURE_KERNEL
# so that the only difference between the two is where the callee lives.
_EXTERNAL_CALLEE_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use helper_state_mod, only : sweep_column").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n")


# A configuration module holding an array, and the kernel that reads it. The
# kernel's own file says nothing about what ``blend_weights`` is, so the
# frontend leaves ``blend_weights(1)`` as a Call: an indexed name in an
# expression is a function reference or an array element, and only the other
# module's source settles which. This is the shape the limited-area kernels
# have, and it is the shape that reaches a TypeError rather than a refusal
# when PSyclone is asked for the callee.
_ARRAY_LIKE_MODULE = """
module weights_config_mod
  use constants_mod, only : r_def
  implicit none
  private
  real(kind=r_def), public :: blend_weights(3) = &
      (/ 1.0_r_def, 2.0_r_def, 3.0_r_def /)
end module weights_config_mod
"""


_ARRAY_LIKE_CALL_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use weights_config_mod, only : blend_weights").replace(
    "      field_out(map_w3(1) + k - 1) = swept(k)",
    "      field_out(map_w3(1) + k - 1) = swept(k) * blend_weights(1)")


# The sweep of _LOCAL_KERNEL taken out into a module procedure of the kernel's
# own module. This is the shape inlining can reach: the callee is a procedure
# of the very Container the call site is in, so no import has to be followed
# and no second module's source has to be read. Its formals are an extent, a
# read column and a written one, which is what a GungHo helper looks like.
_MODULE_PROCEDURE_KERNEL = _LOCAL_KERNEL.replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column\n"
    "end module column_solve_kernel_mod")


# The same module procedure reached through a chain: the kernel calls one
# helper and that helper calls a second. Inlining one call exposes the next,
# so the rewrite has to be repeated until none is left rather than run once.
_CHAINED_PROCEDURE_KERNEL = _MODULE_PROCEDURE_KERNEL.replace(
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n",
    "    call seed_column(n, source, result)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n").replace(
    "  end subroutine sweep_column\n",
    "  end subroutine sweep_column\n"
    "  subroutine seed_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    result(n) = source(n)\n"
    "  end subroutine seed_column\n")


# A callee that calls itself. Inlining it once leaves a call to it behind, so
# a rewrite run to a fixed point would never reach one; the depth limit is
# what turns that into a refusal naming the routine.
_RECURSIVE_PROCEDURE_KERNEL = _MODULE_PROCEDURE_KERNEL.replace(
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n",
    "    result(n) = source(n)\n"
    "    j = n - 1\n"
    "    call sweep_column(j, source, result)\n")


# A kernel handing a column of a rank-2 local to a module procedure, which is
# the shape convert_hdiv_native_code reaches native_jacobian with. The actual
# is a contiguous whole-dimension section, so before inlining it is a section
# outside an assignment; afterwards there is no argument at all, because the
# formal's every use has become a subscript of the local itself.
_SECTION_ACTUAL_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    real(kind=r_def), dimension(nlayers, 2) :: column\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    column(:,1) = partial(:)\n"
    "    call sweep_column(nlayers, column(:,1), swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_def), dimension(n), intent(in) :: source\n"
    "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column\n"
    "end module column_solve_kernel_mod")


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


# The same module variable, assigned to. The region is handed the value the
# module holds when the launch is made and has no share in the module's
# storage, so the assignment would be lost rather than carried back.
_ASSIGNED_MODULE_VARIABLE_KERNEL = _MODULE_VARIABLE_KERNEL.replace(
    "    swept(nlayers) = partial(nlayers)",
    "    profile_size = nlayers\n    swept(nlayers) = partial(nlayers)")


# The same module variable declared `allocatable`. Its shape is not in the
# declaration at all, so there is no View for the region to size and nothing
# the generated interface could state.
_ALLOCATABLE_MODULE_VARIABLE_KERNEL = _MODULE_VARIABLE_KERNEL.replace(
    "  real(kind=r_def), public :: profile_heights(100)",
    "  real(kind=r_def), public, allocatable :: profile_heights(:)")


# A module array whose extent is named rather than stated. The region sizes
# its own View, and it can only size it from what it can evaluate: `n_profile`
# is not one of its arguments.
_SIZED_MODULE_VARIABLE_KERNEL = _MODULE_VARIABLE_KERNEL.replace(
    "  private\n",
    "  private\n"
    "  integer(kind=i_def), public :: n_profile = 100\n").replace(
    "  real(kind=r_def), public :: profile_heights(100)",
    "  real(kind=r_def), public :: profile_heights(n_profile)")


# A variable declared inside the routine with an initialiser, which is
# rtheta_bd_kernel_mod's `upwind`. Fortran gives it the SAVE attribute, so it
# is a static of the routine rather than of the module: nothing outside the
# routine declares it, and the PSy layer has no name to pass.
_LOCAL_STATIC_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    integer(kind=i_def) :: visits = 0\n").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) + visits")


# A module array of a type that has no place on the ABI. A logical crosses by
# conversion, which is per value, so a logical array has nowhere to go -- the
# same refusal a logical array argument meets, reached by a module variable.
_LOGICAL_MODULE_VARIABLE_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, l_def, r_def").replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  logical(kind=l_def), public :: profile_active(100)\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      if (profile_active(k)) swept(k) = swept(k + 1) - partial(k)")


# An array parameter of a type the generated unit has no declaration for. A
# logical crosses the ABI by conversion, which is per value, so an array of
# them has no C type -- and unlike the cases above the values are perfectly
# readable, which is why this refusal is separate from being unable to fold.
_LOGICAL_ARRAY_CONSTANT_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, l_def, r_def").replace(
    "  implicit none",
    "  implicit none\n"
    "  private\n"
    "  logical(kind=l_def), parameter :: sweep(4) = &\n"
    "      [.true., .false., .true., .false.]\n"
    "  public :: column_solve_kernel_type, column_solve_code").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      if (sweep(k)) swept(k) = swept(k + 1) - partial(k)")


# A name reaching the kernel through a wildcard `use`. The import states no
# name, so PSyIR has no container to resolve the symbol in and no declaration
# to type it from: it is not an import the PSy layer could repeat, and not
# anything either module declares as far as the tree can tell.
_WILDCARD_IMPORT_KERNEL = _LOCAL_KERNEL.replace(
    "  use kernel_mod, only : kernel_type",
    "  use kernel_mod, only : kernel_type\n"
    "  use planet_config_mod").replace(
    "      swept(k) = swept(k + 1) - partial(k)",
    "      swept(k) = swept(k + 1) - partial(k) * recip_epsilon")


# The target kernel reading its one imported constant twice. Each reference is
# a separate node, and describing the second would re-resolve a symbol already
# on the ABI and offer the PSy layer the same argument twice.
_REPEATED_IMPORT_KERNEL = _KERNEL.replace(
    "            1.0_r_def + recip_epsilon * mr_v_at_dof",
    "            recip_epsilon + recip_epsilon * mr_v_at_dof")


# The target kernel subscripting one field by a constant its *module* imports,
# on both sides of an assignment. That is the shape LFRic's inter-grid
# prolongation has -- 'fine_field(map_fine(SWB, ...))', with SWB an
# 'integer, parameter' from reference_element_mod -- and nothing about it is
# inter-grid: what matters is that the name is resolved through the kernel
# module's symbol table rather than the subroutine's, and that the loop reads
# and writes one array so that the dependence analysis has two subscripts to
# compare symbolically. 'n_moist' is planet_config_mod's integer parameter.
_MODULE_INDEX_KERNEL = _KERNEL.replace(
    "  use planet_config_mod, only : recip_epsilon",
    "  use planet_config_mod, only : n_moist, recip_epsilon").replace(
    "        moist_dyn_gas(map_wtheta(df) + k) = &\n"
    "            1.0_r_def + recip_epsilon * mr_v_at_dof",
    "        moist_dyn_gas(map_wtheta(n_moist) + k) = &\n"
    "            moist_dyn_gas(map_wtheta(n_moist) + k) + &\n"
    "            recip_epsilon * mr_v_at_dof")


# The target kernel renaming its imported constant, as LFRic's moisture
# kernels rename the latent heats they read. The module declares one name and
# the body reads another, so the generated PSy layer has to repeat the rename
# rather than import either name alone.
_RENAMED_IMPORT_KERNEL = _KERNEL.replace(
    "  use planet_config_mod, only : recip_epsilon",
    "  use planet_config_mod, only : recip => recip_epsilon").replace(
    "            1.0_r_def + recip_epsilon * mr_v_at_dof",
    "            1.0_r_def + recip * mr_v_at_dof")


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


# Face quadrature is a third shape, with a point count and a face count of its
# own. It is outside what this capability models and is refused by name.
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


# A field whose data is integer, and the same kernel over real data beside it.
# LFRic's integer_field_type has field_type's proxy shape and differs in one
# thing, the intrinsic of the data array, so both are instantiated from one
# template: what the generated region does differently for an integer field is
# then read off against a region differing in nothing else. The name is a
# parameter because the two live in one directory and the kernel source is
# reparsed by module name when the transformation resolves the body.
_RATIO_ALGORITHM = """
program kokkos_{name}_test
  use {module}, only : {field}
  use {name}_kernel_mod, only : {name}_kernel_type
  implicit none
  type({field}) :: mask_out, mask_in, mask_div
  call invoke({name}_kernel_type(mask_out, mask_in, mask_div))
end program kokkos_{name}_test
"""


_RATIO_KERNEL = """
module {name}_kernel_mod
  use argument_mod, only : arg_type, gh_field, {intrinsic}, gh_write, &
                           gh_read, cell_column
  use constants_mod, only : i_def, i_native, r_def
  use fs_continuity_mod, only : w3, wtheta
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: {name}_kernel_type
    type(arg_type) :: meta_args(3) = (/                              &
         arg_type(gh_field, {intrinsic}, gh_write, w3),              &
         arg_type(gh_field, {intrinsic}, gh_read,  wtheta),          &
         arg_type(gh_field, {intrinsic}, gh_read,  wtheta) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: {name}_code
  end type {name}_kernel_type
contains
  subroutine {name}_code(nlayers, mask_out, mask_in, mask_div, &
                         ndf_w3, undf_w3, map_w3, &
                         ndf_wtheta, undf_wtheta, map_wtheta)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: ndf_wtheta, undf_wtheta
    {data}, dimension(undf_w3), intent(inout) :: mask_out
    {data}, dimension(undf_wtheta), intent(in) :: mask_in
    {data}, dimension(undf_wtheta), intent(in) :: mask_div
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def), dimension(ndf_wtheta), intent(in) :: map_wtheta
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        mask_out(map_w3(df) + k) = mask_in(map_wtheta(df) + k) &
                                 / mask_div(map_wtheta(df) + k)
      end do
    end do
  end subroutine {name}_code
end module {name}_kernel_mod
"""


_INTEGER_FIELD_ALGORITHM = _RATIO_ALGORITHM.format(
    name="int_ratio", module="integer_field_mod", field="integer_field_type")
_INTEGER_FIELD_KERNEL = _RATIO_KERNEL.format(
    name="int_ratio", intrinsic="gh_integer", data="integer(kind=i_def)")
_REAL_FIELD_ALGORITHM = _RATIO_ALGORITHM.format(
    name="real_ratio", module="field_mod", field="field_type")
_REAL_FIELD_KERNEL = _RATIO_KERNEL.format(
    name="real_ratio", intrinsic="gh_real", data="real(kind=r_def)")


# A field whose data is integer at a kind the ABI has no width for. i_native
# is a real LFRic kind and is deliberately absent from the precision map
# psyclone.cfg carries, so nothing here is monkeypatched: the element type
# following the argument's intrinsic must not become a licence to write 'int'
# for an integer of any width at all.
_OFF_ABI_ALGORITHM = _RATIO_ALGORITHM.format(
    name="native_ratio", module="integer_field_mod",
    field="integer_field_type")
_OFF_ABI_KERNEL = _RATIO_KERNEL.format(
    name="native_ratio", intrinsic="gh_integer",
    data="integer(kind=i_native)")


# One invoke carrying both kinds of field, which is how LFRic actually uses an
# integer one: a mask or an index array read beside the real data it selects.
# The two field types come from two modules, and the region has to give each
# formal its own element type without disturbing the order the PSy layer
# passes them in.
_MIXED_ALGORITHM = """
program kokkos_mixed_test
  use field_mod, only : field_type
  use integer_field_mod, only : integer_field_type
  use masked_copy_kernel_mod, only : masked_copy_kernel_type
  implicit none
  type(field_type) :: out_field, in_field
  type(integer_field_type) :: mask
  call invoke(masked_copy_kernel_type(out_field, in_field, mask))
end program kokkos_mixed_test
"""


_MIXED_KERNEL = """
module masked_copy_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_integer, &
                           gh_write, gh_read, cell_column
  use constants_mod, only : i_def, r_def
  use fs_continuity_mod, only : w3, wtheta
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: masked_copy_kernel_type
    type(arg_type) :: meta_args(3) = (/                              &
         arg_type(gh_field, gh_real,    gh_write, w3),               &
         arg_type(gh_field, gh_real,    gh_read,  w3),               &
         arg_type(gh_field, gh_integer, gh_read,  wtheta) /)
    integer :: operates_on = cell_column
  contains
    procedure, nopass :: masked_copy_code
  end type masked_copy_kernel_type
contains
  subroutine masked_copy_code(nlayers, field_out, field_in, mask, &
                              ndf_w3, undf_w3, map_w3, &
                              ndf_wtheta, undf_wtheta, map_wtheta)
    integer(kind=i_def), intent(in) :: nlayers, ndf_w3, undf_w3
    integer(kind=i_def), intent(in) :: ndf_wtheta, undf_wtheta
    real(kind=r_def), dimension(undf_w3), intent(inout) :: field_out
    real(kind=r_def), dimension(undf_w3), intent(in) :: field_in
    integer(kind=i_def), dimension(undf_wtheta), intent(in) :: mask
    integer(kind=i_def), dimension(ndf_w3), intent(in) :: map_w3
    integer(kind=i_def), dimension(ndf_wtheta), intent(in) :: map_wtheta
    integer(kind=i_def) :: k, df
    do k = 0, nlayers - 1
      do df = 1, ndf_w3
        field_out(map_w3(df) + k) = field_in(map_w3(df) + k) &
                                  * mask(map_wtheta(df) + k)
      end do
    end do
  end subroutine masked_copy_code
end module masked_copy_kernel_mod
"""


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


# A kind-polymorphic kernel that also asks for an evaluator. It is the shape
# GungHo's sample_field_kernel_mod has: the basis is tabulated at the nodal
# points of the written field's space and stays r_def whatever the fields do,
# because the PSy layer computes it once at the model's own working precision.
_POLYMORPHIC_EVALUATOR_MEMBER = """
  subroutine {name}_code_{kind}(nlayers, field_out, field_in,           &
                                ndf_{target}, undf_{target}, map_{target},  &
                                ndf_{space}, undf_{space}, map_{space},     &
                                basis_{space}_on_{target})
    integer(kind=i_def), intent(in) :: nlayers
    integer(kind=i_def), intent(in) :: ndf_{target}, undf_{target}
    integer(kind=i_def), intent(in) :: ndf_{space}, undf_{space}
    integer(kind=i_def), dimension(ndf_{target}), intent(in) :: map_{target}
    integer(kind=i_def), dimension(ndf_{space}), intent(in) :: map_{space}
    real(kind={kind}), dimension(undf_{target}), intent(inout) :: field_out
    real(kind={kind}), dimension(undf_{space}), intent(in) :: field_in
    real(kind=r_def), dimension({dim},ndf_{space},ndf_{target}),           &
                                     intent(in) :: basis_{space}_on_{target}
    integer(kind=i_def) :: k, df, dg
    real(kind={kind}) :: total
    do k = 0, nlayers - 1
      do df = 1, ndf_{target}
        total = 0.0_{kind}
        do dg = 1, ndf_{space}
          total = total + basis_{space}_on_{target}({dim},dg,df)          &
                * field_in(map_{space}(dg) + k)
        end do
        field_out(map_{target}(df) + k) = total
      end do
    end do
  end subroutine {name}_code_{kind}
"""


def _polymorphic_evaluator_kernel(name, first, second, space="w1",
                                  target="w3", dim=3):
    """Build a kernel module whose evaluator code is an interface over kinds.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str first: the kind of the first specific procedure.
    :param str second: the kind of the second.
    :param str space: the function space the basis is asked for on.
    :param str target: the function space of the written field, which is what
        an evaluator tabulates at.
    :param int dim: the basis's first extent, which metadata fixes for a
        named space and leaves unknown for an ``any_space``.

    :returns: Fortran source for the module.
    :rtype: str
    """
    members = "".join(
        _POLYMORPHIC_EVALUATOR_MEMBER.format(
            name=name, kind=kind, space=space, target=target, dim=dim)
        for kind in (first, second))
    # An any_space is named by argument_mod; a named function space by
    # fs_continuity_mod.
    generic = [fs for fs in (space, target) if fs.startswith("any_")]
    named = [fs for fs in (space, target) if not fs.startswith("any_")]
    generic_import = ", " + ", ".join(generic) if generic else ""
    named_import = (f"  use fs_continuity_mod, only : {', '.join(named)}\n"
                    if named else "")
    return f"""
module {name}_kernel_mod
  use argument_mod, only : arg_type, func_type, gh_field, gh_real,        &
                           gh_write, gh_read, gh_basis, cell_column,      &
                           gh_evaluator{generic_import}
  use constants_mod, only : i_def, r_def, r_single, r_double, r_solver
{named_import}  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: {name}_kernel_type
    type(arg_type) :: meta_args(2) = (/                                  &
         arg_type(gh_field, gh_real, gh_write, {target}),                 &
         arg_type(gh_field, gh_real, gh_read,  {space}) /)
    type(func_type) :: meta_funcs(1) = (/                                &
         func_type({space}, gh_basis) /)
    integer :: gh_shape = gh_evaluator
    integer :: operates_on = cell_column
  end type {name}_kernel_type
  public :: {name}_code
  interface {name}_code
    module procedure {name}_code_{first}, {name}_code_{second}
  end interface
contains
{members}end module {name}_kernel_mod
"""


def _polymorphic_evaluator_algorithm(name, field_module, field_type):
    """Build an algorithm invoking one polymorphic evaluator kernel.

    An evaluator carries no quadrature object, so the invoke passes nothing
    but the two fields; their precision is the whole of what selects a member.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str field_module: the module the field type comes from.
    :param str field_type: the LFRic field type, whose precision selects the
        specific procedure.

    :returns: Fortran source for the program.
    :rtype: str
    """
    return f"""
program kokkos_{name}_test
  use {field_module}, only : {field_type}
  use {name}_kernel_mod, only : {name}_kernel_type
  implicit none
  type({field_type}) :: out_field, in_field
  call invoke({name}_kernel_type(out_field, in_field))
end program kokkos_{name}_test
"""


# A kind-polymorphic inter-grid kernel, the shape the model's
# sci_restrict_scalar_unweighted_kernel_mod has: one restriction written twice
# over, once per real kind, behind a generic interface.
_POLYMORPHIC_INTERGRID_MEMBER = """
  subroutine {name}_code_{kind}(nlayers, cell_map, ncell_f_per_c_x,      &
                                ncell_f_per_c_y, ncell_f, coarse_field,  &
                                fine_field, undf_c, map_c, ndf, undf_f,  &
                                map_f)
    integer(kind=i_def), intent(in) :: nlayers, ncell_f_per_c_x
    integer(kind=i_def), intent(in) :: ncell_f_per_c_y, ncell_f
    integer(kind=i_def), intent(in) :: ndf, undf_f, undf_c
    integer(kind=i_def), dimension(ncell_f_per_c_x, ncell_f_per_c_y), &
        intent(in) :: cell_map
    real(kind={kind}), dimension(undf_c), intent(inout) :: coarse_field
    real(kind={kind}), dimension(undf_f), intent(in) :: fine_field
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
  end subroutine {name}_code_{kind}
"""


def _polymorphic_intergrid_kernel(name, first, second):
    """Build an inter-grid kernel module whose code is an interface.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str first: the kind of the first specific procedure.
    :param str second: the kind of the second.

    :returns: Fortran source for the module.
    :rtype: str
    """
    members = "".join(
        _POLYMORPHIC_INTERGRID_MEMBER.format(name=name, kind=kind)
        for kind in (first, second))
    return f"""
module {name}_kernel_mod
  use argument_mod, only : arg_type, gh_field, gh_real, gh_readwrite,      &
                           gh_read, gh_coarse, gh_fine, cell_column,       &
                           any_discontinuous_space_1,                      &
                           any_discontinuous_space_2
  use constants_mod, only : i_def, r_def, r_single, r_double, r_solver
  use kernel_mod, only : kernel_type
  implicit none
  type, public, extends(kernel_type) :: {name}_kernel_type
    type(arg_type) :: meta_args(2) = (/                                    &
         arg_type(gh_field, gh_real, gh_readwrite,                         &
                  any_discontinuous_space_1, mesh_arg=gh_coarse),          &
         arg_type(gh_field, gh_real, gh_read,                              &
                  any_discontinuous_space_2, mesh_arg=gh_fine) /)
    integer :: operates_on = cell_column
  end type {name}_kernel_type
  public :: {name}_code
  interface {name}_code
    module procedure {name}_code_{first}, {name}_code_{second}
  end interface
contains
{members}end module {name}_kernel_mod
"""


def _polymorphic_intergrid_algorithm(name, field_module, field_type):
    """Build an algorithm invoking one polymorphic inter-grid kernel.

    :param str name: the kernel's base name, without ``_kernel_mod``.
    :param str field_module: the module the field type comes from.
    :param str field_type: the LFRic field type, whose precision selects the
        specific procedure.

    :returns: Fortran source for the program.
    :rtype: str
    """
    return f"""
program kokkos_{name}_test
  use {field_module}, only : {field_type}
  use {name}_kernel_mod, only : {name}_kernel_type
  implicit none
  type({field_type}) :: coarse_field, fine_field
  call invoke({name}_kernel_type(coarse_field, fine_field))
end program kokkos_{name}_test
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


@pytest.fixture(name="integer_field_target")
# pylint: disable-next=unused-argument
def integer_field_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose every field carries integer data."""
    return _invoke(
        tmp_path, "int_ratio", _INTEGER_FIELD_ALGORITHM,
        _INTEGER_FIELD_KERNEL)


@pytest.fixture(name="real_field_target")
# pylint: disable-next=unused-argument
def real_field_target_fixture(tmp_path, clear_module_manager_instance):
    """Create the same invoke over real fields, to read the other against."""
    return _invoke(
        tmp_path, "real_ratio", _REAL_FIELD_ALGORITHM, _REAL_FIELD_KERNEL)


@pytest.fixture(name="off_abi_field_target")
# pylint: disable-next=unused-argument
def off_abi_field_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose integer fields are of a kind off the ABI."""
    return _invoke(
        tmp_path, "native_ratio", _OFF_ABI_ALGORITHM, _OFF_ABI_KERNEL)


@pytest.fixture(name="mixed_field_target")
# pylint: disable-next=unused-argument
def mixed_field_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke taking real fields and an integer one together."""
    return _invoke(
        tmp_path, "masked_copy", _MIXED_ALGORITHM, _MIXED_KERNEL)


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


@pytest.fixture(name="coloured_target")
# pylint: disable-next=unused-argument
def coloured_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose captured loop has a continuous-space sibling."""
    return _invoke(
        tmp_path, "scaled_copy", _COLOURED_ALGORITHM, _SECOND_KERNEL,
        extra={"assemble_w2_kernel_mod": _COLOURED_KERNEL})


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


@pytest.fixture(name="default_logical_target")
# pylint: disable-next=unused-argument
def default_logical_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel takes a logical declared with no kind."""
    return _invoke(
        tmp_path, "masked_solver", _LOGICAL_ALGORITHM,
        _DEFAULT_LOGICAL_KERNEL)


@pytest.fixture(name="default_integer_target")
# pylint: disable-next=unused-argument
def default_integer_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel takes seven default-kind integers."""
    return _invoke(
        tmp_path, "average_w3_to_w0", _DEFAULT_INTEGER_ALGORITHM,
        _DEFAULT_INTEGER_KERNEL)


@pytest.fixture(name="default_real_target")
# pylint: disable-next=unused-argument
def default_real_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel takes a real declared with no kind."""
    return _invoke(
        tmp_path, "scaled_solver", _SOLVER_ALGORITHM, _DEFAULT_REAL_KERNEL)


@pytest.fixture(name="default_constant_target")
# pylint: disable-next=unused-argument
def default_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads an unkinded module constant."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _DEFAULT_CONSTANT_KERNEL)


@pytest.fixture(name="operator_target")
# pylint: disable-next=unused-argument
def operator_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads an LMA operator."""
    return _invoke(
        tmp_path, "dg_matrix_vector", _OPERATOR_ALGORITHM, _OPERATOR_KERNEL)


@pytest.fixture(name="two_operator_target")
# pylint: disable-next=unused-argument
def two_operator_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads two LMA operators."""
    return _invoke(
        tmp_path, "dg_matrix_vector", _TWO_OPERATOR_ALGORITHM,
        _TWO_OPERATOR_KERNEL)


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


@pytest.fixture(name="polymorphic_intergrid_target")
# pylint: disable-next=unused-argument
def polymorphic_intergrid_target_fixture(tmp_path,
                                         clear_module_manager_instance):
    """Create an invoke of a kind-polymorphic inter-grid kernel.

    As for ``polymorphic_target``, the r_double member is declared first and
    the algorithm passes r_solver fields, so the match is the second member.
    """
    return _invoke(
        tmp_path, "restrict_poly",
        _polymorphic_intergrid_algorithm(
            "restrict_poly", "r_solver_field_mod", "r_solver_field_type"),
        _polymorphic_intergrid_kernel("restrict_poly", "r_double", "r_single"))


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


@pytest.fixture(name="face_quadrature_target")
# pylint: disable-next=unused-argument
def face_quadrature_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads face-quadrature basis data."""
    return _invoke(
        tmp_path, "face_weight", _FACE_QUADRATURE_ALGORITHM,
        _FACE_QUADRATURE_KERNEL)


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


@pytest.fixture(name="implicit_target")
# pylint: disable-next=unused-argument
def implicit_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel leaves two formals' extents out."""
    return _invoke(
        tmp_path, "normals_sum", _IMPLICIT_ALGORITHM, _IMPLICIT_KERNEL)


@pytest.fixture(name="implicit_bound_target")
# pylint: disable-next=unused-argument
def implicit_bound_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel asks an assumed shape for its bounds."""
    return _invoke(
        tmp_path, "normals_sum", _IMPLICIT_ALGORITHM, _IMPLICIT_BOUND_KERNEL)


@pytest.fixture(name="implicit_origin_target")
# pylint: disable-next=unused-argument
def implicit_origin_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel states one bound of an assumed shape."""
    return _invoke(
        tmp_path, "normals_sum", _IMPLICIT_ALGORITHM, _IMPLICIT_ORIGIN_KERNEL)


@pytest.fixture(name="implicit_local_target")
# pylint: disable-next=unused-argument
def implicit_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local of no stated shape."""
    return _invoke(
        tmp_path, "normals_sum", _IMPLICIT_ALGORITHM, _IMPLICIT_LOCAL_KERNEL)


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


@pytest.fixture(name="zero_based_local_target")
# pylint: disable-next=unused-argument
def zero_based_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a local from zero."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ZERO_BASED_LOCAL_KERNEL)


@pytest.fixture(name="zero_based_enquiry_target")
# pylint: disable-next=unused-argument
def zero_based_enquiry_target_fixture(tmp_path,
                                      clear_module_manager_instance):
    """Create an invoke asking all three enquiries of a zero-based local."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _ZERO_BASED_ENQUIRY_KERNEL)


@pytest.fixture(name="negative_origin_local_target")
# pylint: disable-next=unused-argument
def negative_origin_local_target_fixture(tmp_path,
                                         clear_module_manager_instance):
    """Create an invoke whose kernel centres a local on zero."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _NEGATIVE_ORIGIN_LOCAL_KERNEL)


@pytest.fixture(name="unrenderable_extent_target")
# pylint: disable-next=unused-argument
def unrenderable_extent_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create an invoke whose kernel squares in a declared upper bound."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _UNRENDERABLE_EXTENT_KERNEL)


@pytest.fixture(name="squared_extent_target")
# pylint: disable-next=unused-argument
def squared_extent_target_fixture(tmp_path,
                                  clear_module_manager_instance):
    """Create an invoke whose kernel squares a name in a declared extent."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _SQUARED_EXTENT_KERNEL)


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


@pytest.fixture(name="shapeless_local_target")
# pylint: disable-next=unused-argument
def shapeless_local_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel allocates a local in its body."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _SHAPELESS_LOCAL_KERNEL)


@pytest.fixture(name="unwritable_shape_target")
# pylint: disable-next=unused-argument
def unwritable_shape_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel takes a MAX in a declared bound."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _UNWRITABLE_SHAPE_KERNEL)


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


@pytest.fixture(name="module_procedure_target")
# pylint: disable-next=unused-argument
def module_procedure_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel calls a procedure of its own module."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _MODULE_PROCEDURE_KERNEL)


@pytest.fixture(name="chained_procedure_target")
# pylint: disable-next=unused-argument
def chained_procedure_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel's callee itself calls."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _CHAINED_PROCEDURE_KERNEL)


@pytest.fixture(name="recursive_procedure_target")
def recursive_procedure_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create an invoke whose kernel calls a self-recursive procedure."""
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _RECURSIVE_PROCEDURE_KERNEL)


@pytest.fixture(name="section_actual_target")
# pylint: disable-next=unused-argument
def section_actual_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke passing a contiguous section to a module procedure."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _SECTION_ACTUAL_KERNEL)


@pytest.fixture(name="external_callee_target")
# pylint: disable-next=unused-argument
def external_callee_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose callee is readable but in another module."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _EXTERNAL_CALLEE_KERNEL,
        extra={"helper_state_mod": _EXTERNAL_STATE_MODULE})


@pytest.fixture(name="array_like_call_target")
# pylint: disable-next=unused-argument
def array_like_call_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke reading an array the frontend takes for a call."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _ARRAY_LIKE_CALL_KERNEL,
        extra={"weights_config_mod": _ARRAY_LIKE_MODULE})


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


@pytest.fixture(name="module_variable_target")
# pylint: disable-next=unused-argument
def module_variable_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads public state of its own module."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _MODULE_VARIABLE_KERNEL)


@pytest.fixture(name="assigned_module_variable_target")
# pylint: disable-next=unused-argument
def assigned_module_variable_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel assigns to state of its own module."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _ASSIGNED_MODULE_VARIABLE_KERNEL)


@pytest.fixture(name="allocatable_module_variable_target")
# pylint: disable-next=unused-argument
def allocatable_module_variable_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads an allocatable module array."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _ALLOCATABLE_MODULE_VARIABLE_KERNEL)


@pytest.fixture(name="sized_module_variable_target")
# pylint: disable-next=unused-argument
def sized_module_variable_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads a module array of named extent."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _SIZED_MODULE_VARIABLE_KERNEL)


@pytest.fixture(name="local_static_target")
# pylint: disable-next=unused-argument
def local_static_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a routine-local static."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _LOCAL_STATIC_KERNEL)


@pytest.fixture(name="logical_array_constant_target")
# pylint: disable-next=unused-argument
def logical_array_constant_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a logical array parameter."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _LOGICAL_ARRAY_CONSTANT_KERNEL)


@pytest.fixture(name="logical_module_variable_target")
# pylint: disable-next=unused-argument
def logical_module_variable_target_fixture(
        tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads a module logical array."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _LOGICAL_MODULE_VARIABLE_KERNEL)


@pytest.fixture(name="wildcard_import_target")
# pylint: disable-next=unused-argument
def wildcard_import_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads a name from a wildcard import."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _WILDCARD_IMPORT_KERNEL,
        extra={"planet_config_mod": _PLANET_CONFIG})


@pytest.fixture(name="repeated_import_target")
# pylint: disable-next=unused-argument
def repeated_import_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads one imported constant twice."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _REPEATED_IMPORT_KERNEL)


@pytest.fixture(name="module_index_target")
# pylint: disable-next=unused-argument
def module_index_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel indexes a field by a module constant."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _MODULE_INDEX_KERNEL)


@pytest.fixture(name="reshaped_constant_target")
# pylint: disable-next=unused-argument
def reshaped_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel reads a reshaped parameter array."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _RESHAPED_CONSTANT_KERNEL)


@pytest.fixture(name="folded_constant_target")
# pylint: disable-next=unused-argument
def folded_constant_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel declares a parameter as an expression."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _FOLDED_CONSTANT_KERNEL,
        extra={"planet_config_mod": _PLANET_CONFIG})


@pytest.fixture(name="renamed_import_target")
# pylint: disable-next=unused-argument
def renamed_import_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel renames the constant it imports."""
    return _invoke(
        tmp_path, "moist_dyn_gas", _ALGORITHM, _RENAMED_IMPORT_KERNEL)


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


@pytest.fixture(name="polymorphic_evaluator_target")
# pylint: disable-next=unused-argument
def polymorphic_evaluator_target_fixture(tmp_path,
                                         clear_module_manager_instance):
    """Create an invoke of a polymorphic interface asking for an evaluator.

    The precisions are ``polymorphic_target``'s, so the selection question is
    the one that already has an answer; what is new is the ``gh_evaluator``
    metadata standing between the matcher and it. The r_double member is
    declared first, so a selection returning ``schedules[0]`` returns the
    wrong one.
    """
    return _invoke(
        tmp_path, "eval_scale",
        _polymorphic_evaluator_algorithm(
            "eval_scale", "r_solver_field_mod", "r_solver_field_type"),
        _polymorphic_evaluator_kernel("eval_scale", "r_double", "r_single"))


@pytest.fixture(name="any_space_evaluator_target")
# pylint: disable-next=unused-argument
def any_space_evaluator_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create a polymorphic evaluator interface on an ``any_space``.

    Metadata cannot say how many components a basis on an ``any_space`` has,
    which is PSyclone's issue #461, so the interface the matcher builds cannot
    state the basis's first extent. That extent is never compared, so it must
    not be what stops the question being asked.
    """
    return _invoke(
        tmp_path, "any_scale",
        _polymorphic_evaluator_algorithm(
            "any_scale", "r_solver_field_mod", "r_solver_field_type"),
        _polymorphic_evaluator_kernel("any_scale", "r_double", "r_single",
                                      space="any_space_2",
                                      target="any_space_1", dim=1))


@pytest.fixture(name="ambiguous_evaluator_target")
# pylint: disable-next=unused-argument
def ambiguous_evaluator_target_fixture(tmp_path,
                                       clear_module_manager_instance):
    """Create an evaluator interface every member of which matches.

    ``precision_map`` gives r_single and r_solver the same 4 bytes. The point
    of asking it again over evaluator metadata is that the ambiguity must
    survive the interface becoming buildable: a matcher that now answers must
    answer "both" here rather than "the first one".
    """
    return _invoke(
        tmp_path, "dual_eval",
        _polymorphic_evaluator_algorithm(
            "dual_eval", "r_solver_field_mod", "r_solver_field_type"),
        _polymorphic_evaluator_kernel("dual_eval", "r_single", "r_solver"))


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


def test_lfric_kokkos_trans_defers_a_section_actual_to_the_call(
        tmp_path, clear_module_manager_instance):
    """A section given to a routine is refused for the routine, not the shape.

    ``call native_jacobian(vector(:,1))`` carries two reasons a capture
    cannot proceed: the section stands outside an assignment, and the callee
    cannot be inlined. Only the second is the loop's real blocker -- inline
    the callee and the argument, section and all, goes with it -- so the
    section rule steps aside and lets the inlining rule answer. The coverage
    survey asks each rule on its own and keeps every message, so a rule that
    answered here would report this loop as an array-section blocker that no
    array-section work could ever clear.
    """
    # pylint: disable=unused-argument
    _, loop, _ = _invoke(
        tmp_path, "fv_difference", _SECTION_ALGORITHM, _HDIV_CALL_KERNEL)

    # Asked on its own, as the survey asks it, the section rule has nothing
    # to say about this kernel: every assignment in it lowers, and the one
    # section that does not is the call's argument.
    LFRicKokkosTrans._validate_sections(
        LFRicKokkosTrans._schedule(loop.kernels()[0]))

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "cannot inline the call to 'native_jacobian'" in str(error.value)
    assert "array section outside an assignment" not in str(error.value)


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


def test_lfric_kokkos_trans_refuses_an_unhandled_eval_shape(
        face_quadrature_target):
    """A shape outside the two modelled ones is refused by name.

    Face and edge quadrature carry a face count and a single point count
    rather than the XYoZ pair. Nothing here has been measured against the
    model for them, so the refusal stays and says which shape it is about
    rather than reporting quadrature as a whole as unsupported.
    """
    _, loop, kernel = face_quadrature_target
    assert kernel.eval_shapes == ["gh_quadrature_face"]

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("LFRicKokkosTrans does not support the 'gh_quadrature_face' "
            "evaluator shape." in str(error.value))


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


def test_lfric_kokkos_trans_selects_an_intergrid_schedule(
        polymorphic_intergrid_target):
    """A kind-polymorphic inter-grid kernel is selected, not refused.

    Selecting between an interface's members means building the argument list
    the metadata implies and comparing it with each member's, and PSyclone's
    issue #928 left the two inter-grid callbacks of that builder unwritten.
    Every polymorphic inter-grid kernel in the model was therefore refused for
    a reason that had nothing to do with what a region can express -- the
    question could not be asked. Asking it is what this asserts, and the
    r_single answer is what says the comparison really ran rather than
    returning the first member.
    """
    _, loop, kernel = polymorphic_intergrid_target
    assert kernel.is_intergrid
    assert len(kernel.get_callees()) == 2

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert schedule.name == "restrict_poly_code_r_single"
    assert schedule is not kernel.get_callees()[0]

    code = LFRicKokkosTrans().apply(loop)
    assert "double" not in code
    assert "float *coarse_field_data" in code
    assert "const float *fine_field_data" in code
    assert "const int *cell_map_data" in code


def test_lfric_kokkos_trans_rejects_a_columnwise_assembly(target):
    """The CMA refusal is by operation, ahead of the argument-type walk.

    A CMA kernel is refused for what it does rather than for what it takes:
    an assembly kernel builds a banded matrix from an LMA one, so its
    arguments alone would now pass. The two refusals are therefore both
    needed and are asserted apart.
    """
    _, loop, kernel = target
    kernel._cma_operation = "assembly"
    with pytest.raises(TransformationError, match="CMA operators"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_rejects_a_columnwise_operator(target):
    """A CMA operator is refused where an LMA one is now accepted.

    The two are not variants of one capability. An LMA operator is a rank-3
    array the kernel slices by cell, which the region describes as a View like
    any other; a CMA operator is a banded matrix with its own bandwidth and
    indexing arguments, and none of that machinery exists here.
    """
    _, loop, kernel = target
    kernel.arguments.args[1]._argument_type = "gh_columnwise_operator"
    with pytest.raises(TransformationError, match="gh_columnwise_operator"):
        LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_accepts_an_operator(operator_target):
    """An LMA operator becomes a rank-3 read-only View and nothing else.

    Every extent of the local stencil -- ncell_3d, ndf1 and ndf2 -- is itself
    a kernel formal, so the operator needs no machinery of its own: the
    existing View description covers it.
    """
    _, loop, _ = operator_target
    code = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::View<const double***, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> matrix(matrix_data, ncell_3d, ndf1, ndf2);" in code
    assert "const double *matrix_data" in code
    assert "const int ncell_3d" in code


def test_lfric_kokkos_trans_declares_the_cell_position(operator_target):
    """The kernel's cell argument is declared, not taken across the ABI.

    LFRic passes its loop counter for that argument, and the counter is
    lowered away when the loop becomes a launch. Taking it would therefore
    read an unassigned variable on every cell -- a wrong answer rather than a
    build failure, which is why this is asserted from both sides.
    """
    _, loop, _ = operator_target
    code = LFRicKokkosTrans().apply(loop)

    # The launch renames its own index, because the kernel has taken 'cell'.
    assert "const int cell = cell_1 + 1;" in code

    signature = code.split(") {\n")[0]
    parameters = [
        parameter.strip().split()[-1] for parameter in signature.split(",")
        if parameter.strip().split()
    ]
    assert "cell" not in parameters
    assert "ncells" in parameters


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
        assert f"Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, " \
            f"ReadOnly> {dofmap}({dofmap}_data, ndf" in code
        assert f"{dofmap}_data, ndf1, ncells)" in code or \
            f"{dofmap}_data, ndf2, ncells)" in code
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

    assert "matrix(matrix_data, ncell_3d, ndf1, ndf2);" in code
    assert "matrix2(matrix2_data, ncell_3d_2, ndf1, ndf2);" in code
    assert code.count("const int cell = cell_1 + 1;") == 1


def test_lfric_kokkos_trans_operator_cell_actual_must_be_the_counter(
        operator_target, monkeypatch):
    """A cell actual that is not the loop's own variable is refused.

    The transformation reads the position of the cell argument from
    ``has_operator()``, mirroring ``ArgOrdering.generate``. If those two ever
    part company the actual at that index stops being the loop variable, and
    the region would drop the wrong argument -- so the assumption is checked
    rather than trusted.
    """
    _, loop, _ = operator_target
    original = KernCallArgList.generate

    def _shuffled(self, var_accesses=None):
        original(self, var_accesses=var_accesses)
        # Put something that is not the loop variable where the cell is.
        self._psyir_arglist[0] = self._psyir_arglist[1].copy()

    monkeypatch.setattr(KernCallArgList, "generate", _shuffled)
    with pytest.raises(TransformationError, match="cell index"):
        LFRicKokkosTrans().apply(loop)


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

    assert ("Kokkos::View<const int*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> smap_size(smap_size_data, ncells);" in cpp)
    assert "const int *smap_size_data" in cpp
    # '\b' stops either pattern reaching 'smap_size_data' or 'smap_size_max',
    # so what is left is the size itself: never bare, and subscripted by the
    # cell everywhere but the View construction.
    assert not re.findall(r"\bsmap_size\b(?!\()", cpp)
    assert set(re.findall(r"\bsmap_size\(([^)]*)\)", cpp)) == {
        "smap_size_data, ncells", "cell"}

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

    assert ("Kokkos::View<const int***, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> smap(smap_data, ndf_w3, smap_size_max, ncells);"
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

    assert ("Kokkos::View<const int*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> smap_size(smap_size_data, ncells);" in cpp)
    assert ("Kokkos::View<const int***, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> smap(smap_data, ndf_w3, smap_size_max, ncells);"
            in cpp)
    assert "for(step=1; step<=smap_size(cell); step+=1)" in cpp

    fortran = str(psy.gen)
    assert "STENCIL_CROSS, extent" in fortran
    assert fortran.index("halo_exchange(depth=extent)") < fortran.index(
        "call stencil_line_kokkos(")


def test_lfric_kokkos_trans_sizes_an_assumed_shape_from_the_actual(
        implicit_target):
    """A formal declared '(:)' is sized by the array the PSy layer passes.

    The declaration states no extent, but the extent is not unknown: Fortran
    takes it from the actual at the call, and the PSy layer holds that array.
    So the region carries the measurement -- 'SIZE' of the actual -- as a
    scalar of its own and sizes the View from that, which is the same route
    C2 measures a stencil dofmap's storage extent by.

    The kernel's own 'size(adjacent_face, 1)' has to resolve to that same
    scalar. A View sized by one reading of the shape and a loop bounded by
    another is a wrong answer rather than a refusal, which is why the two are
    asserted together.
    """
    psy, loop, _ = implicit_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert ("Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> adjacent_face(adjacent_face_data, "
            "adjacent_face_extent_1, ncells);" in cpp)
    assert "const int adjacent_face_extent_1" in cpp
    assert "for(face=1; face<=adjacent_face_extent_1; face+=1)" in cpp

    fortran = str(psy.gen)
    assert "integer(c_int), value :: adjacent_face_extent_1" in fortran
    assert "SIZE(adjacent_face, dim=1)" in fortran


def test_lfric_kokkos_trans_sizes_a_rank_2_assumed_shape(implicit_target):
    """Every dimension left out of the declaration is measured, in order.

    A rank-2 assumed shape carries two extents and both are the actual's, so
    the region takes two scalars and the PSy layer measures the same array
    twice. The call is asserted whole because the scalars are positional: two
    measurements of one array in the wrong order size the View wrongly in a
    way that still compiles.
    """
    psy, loop, _ = implicit_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert ("Kokkos::View<const double**, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> outward_normals(outward_normals_data, "
            "outward_normals_extent_1, outward_normals_extent_2);" in cpp)
    assert "const int outward_normals_extent_1" in cpp
    assert "const int outward_normals_extent_2" in cpp

    fortran = str(psy.gen)
    assert ("call normals_sum_kokkos(nlayers_field_out, field_out_data, "
            "field_in_data, ndf_w3, undf_w3, map_w3, nfaces_re_h, "
            "out_normals_to_horiz_faces, adjacent_face, "
            "SIZE(out_normals_to_horiz_faces, dim=1), "
            "SIZE(out_normals_to_horiz_faces, dim=2), "
            "SIZE(adjacent_face, dim=1), loop0_stop)" in fortran)


def test_lfric_kokkos_trans_refuses_an_assumed_shape_with_no_actual(
        implicit_local_target):
    """An assumed shape is only measurable where there is a call to measure.

    A kernel-local array declared '(:)' has no actual anywhere, so there is
    nothing to read its extent from and the region refuses it rather than
    choosing one. The refusal names the array and says which of the two
    shapeless declarations it carries, so a reader knows whether to look at
    the kernel or at its caller.
    """
    _, loop, _ = implicit_local_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires 'loose' to be declared with explicit bounds, but it is "
            "declared with an assumed shape" in str(error.value))


def test_lfric_kokkos_trans_refuses_an_assumed_shape_with_a_stated_origin(
        implicit_origin_target):
    """A shape half declared and half measured is not read from two places.

    'dimension(0:)' takes its extent from the actual and its origin from the
    declaration. Honouring both would give the View a shape assembled out of
    the caller and the callee at once, which is the confusion the declared
    bounds exist to avoid, so the measurement is not attempted and the
    refusal says which of the two it found.
    """
    _, loop, _ = implicit_origin_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires 'adjacent_face' to be declared with explicit bounds, "
            "but it is declared with an assumed shape whose lower bound its "
            "declaration states, so its origin and its size would be read "
            "from two different places" in str(error.value))


def test_lfric_kokkos_trans_assumed_shape_lower_bound_is_one(
        implicit_bound_target):
    """An assumed-shape formal is 1-based whatever the actual was declared as.

    Fortran gives the dummy the actual's *extent* and its own lower bound,
    which is 1 unless the dummy states otherwise. The origin A3 shifts every
    subscript by is therefore the declaration's own and must not be taken from
    the actual: 'lbound' is the literal 1, 'ubound' is the measured extent,
    and the subscripts are shifted by one like any other 1-based array's.
    """
    _, loop, _ = implicit_bound_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "for(face=1; face<=adjacent_face_extent_1; face+=1)" in cpp
    assert "adjacent_face((face - 1), cell)" in cpp


def test_lfric_kokkos_trans_refuses_an_unsupported_stencil_type(target):
    """Stencil storage and halo requirements are not silently captured.

    The refusal is shape-specific: 'cross', 'cross2d' and 'region' are
    accepted, and every other shape -- 'xory1d' here, which has a direction
    argument on top of a 1-D size -- is refused by a message naming both the
    shape it found and the whole set it would have taken.
    """
    _, loop, kernel = target
    kernel.arguments.args[1].stencil = LFRicArgStencil(name="xory1d")

    with pytest.raises(TransformationError, match="stencil") as error:
        LFRicKokkosTrans().validate(loop)

    assert "'xory1d'" in str(error.value)
    for shape in LFRicKokkosTrans._SUPPORTED_STENCILS:
        assert shape in str(error.value)


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
    """The fixed C ABI must fail closed if a Fortran kind changes.

    The witness is an integer whose declaration states a width rather than a
    kind name -- 'integer*8', as PSyIR records it. Stage 5 admits an integer
    that states neither, reading it as the default kind and asserting that
    width against the compiler; this one is not that, because the declaration
    did say which width it wanted and it is not the ABI's. Reading the two the
    same way would drop the top four bytes of every value in silence, so the
    pair is tested rather than only the half that is admitted.
    """
    _, loop, kernel = target
    schedule = kernel.get_callees()[0]
    schedule.symbol_table.lookup("nlayers").datatype = ScalarType(
        ScalarType.Intrinsic.INTEGER, 8)
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


def test_lfric_kokkos_trans_accepts_a_default_kind_logical(
        default_logical_target):
    """A logical declared with no kind crosses exactly as a kinded one does.

    LFRic's l_def is kind(.false.), so 'logical' and 'logical(kind=l_def)'
    name the same type and GungHo writes both. The conversion the ABI already
    uses for a logical is what makes the unnamed kind cost nothing: the dummy
    is logical(c_bool), value and the call site wraps the actual, so nothing
    on either side has to know what width the kernel's declaration meant.

    The absence of an assertion is therefore part of the result, not an
    omission from it. A width that is never read is a width that cannot be
    wrong, and asserting one here would invent a claim the generated code does
    not make.
    """
    psy, loop, _ = default_logical_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const bool masked" in cpp
    assert "logical(c_bool), value :: masked" in fortran
    assert "LOGICAL(masked, kind=c_bool)" in fortran
    # No width is asserted for it, and it contributes no name to the assertion
    # block: the kinds asserted are the two the kernel still names.
    assert "storage_size(.true." not in fortran
    assert "use constants_mod, only : i_def, r_solver" in fortran


def test_lfric_kokkos_trans_accepts_default_kind_integers(
        default_integer_target):
    """Default-kind integers cross as int, at a width the compiler checks.

    All seven of the integers LFRic's argument ordering supplies are declared
    with no kind here, as average_w3_to_w0_code declares them. An integer does
    cross the ABI as a width, so unlike the logical above this one cannot be
    admitted by ignoring the question: it is admitted by asking the compiler
    instead of the precision map, which has no entry to be asked about because
    a default kind is precisely the one with no name to key it by.

    The assertion is that question. It compares storage_size of a default
    literal against storage_size of a c_int one, in the generated interface,
    so a build whose default integer is not c_int fails to compile rather than
    reading four bytes where the Fortran wrote eight.
    """
    psy, loop, _ = default_integer_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const int nlayers" in cpp
    assert "const int ndf_w0" in cpp
    assert "const int *map_w0_data" in cpp
    assert "Kokkos::View<const int**, Kokkos::LayoutLeft" in cpp
    for name in ("nlayers", "ndf_w3", "undf_w3", "ndf_w0", "undf_w0"):
        assert f"integer(c_int), value :: {name}" in fortran
    for name in ("map_w3", "map_w0"):
        assert f"integer(c_int), dimension(*), intent(in) :: {name}" in fortran

    # The width is measured rather than assumed, and this is the measurement.
    # The probe is the bare literal, because the kind it is about is the one a
    # literal has when nothing is said about it.
    assert ("integer(kind=merge(4, -1, storage_size(1) == &\n"
            "        storage_size(1_c_int))), parameter :: "
            "assert_kind_default_integer = 0") in fortran
    # constants_mod is asked only for the kind the kernel still names. The
    # default kind is declared nowhere -- being unnamed is what makes it the
    # default -- so importing it by that name would not compile.
    assert "use constants_mod, only : r_def\n" in fortran
    assert "only : default_integer" not in fortran


def test_lfric_kokkos_trans_rejects_a_default_kind_real(default_real_target):
    """A real declared with no kind stays off the ABI.

    The integer above is admitted because LFRic has one default integer and
    means it. It has no default real: r_def, r_solver, r_single and r_tran are
    all in use and all different, so the kind is the whole of what a real
    declaration says about its width. Reading a missing one as 'whatever the
    compiler picks' would put a promotion or a truncation on the ABI silently,
    which is the failure every other refusal here exists to prevent.
    """
    _, loop, _ = default_real_target

    with pytest.raises(TransformationError, match="argument kinds") as error:
        LFRicKokkosTrans().validate(loop)

    assert "'scaling'" in str(error.value)


def test_lfric_kokkos_trans_accepts_an_unkinded_module_constant(
        default_constant_target):
    """The declaration reader admits an unkinded constant on the same terms.

    A module constant carrying PROTECTED leaves the frontend with the original
    text rather than a typed symbol, so its kind is recovered by reading the
    declaration. That reader has to answer the unkinded case the same way the
    typed path does, or the contract would depend on which attributes a
    constant happens to carry: 'quenching' is a plain logical, and it crosses
    by the same conversion 'rehabilitate' does.
    """
    psy, loop, _ = default_constant_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const bool quenching" in cpp
    assert "if (quenching)" in cpp
    assert "LOGICAL(quenching, kind=c_bool)" in fortran


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


def test_lfric_kokkos_trans_accepts_a_polymorphic_kernel(
        polymorphic_evaluator_target):
    """An interface asking for an evaluator is selected between, not refused.

    Until this task the matcher could not build the interface an evaluator
    implies -- PSyclone's issue #928 -- so every member of such an interface
    was refused together, whatever its precisions were. The two members here
    differ in exactly the way ``polymorphic_target``'s do, so the answer is
    the same answer; the only new thing is that it can now be reached.

    The assertions are about the width rather than about a region existing,
    because the r_double member is declared first: returning ``schedules[0]``
    would generate a region that compiles and is wrong. The basis stays
    ``double`` in both members, which is what LFRic writes -- the PSy layer
    tabulates it once at r_def -- so the region carries a float field beside
    a double basis and that mixture is itself the evidence that the formals
    were read rather than assumed.
    """
    _, loop, kernel = polymorphic_evaluator_target
    assert kernel.eval_shapes == ["gh_evaluator"]
    assert len(kernel.get_callees()) == 2

    # pylint: disable-next=protected-access
    schedule = LFRicKokkosTrans._schedule(kernel)
    assert schedule.name == "eval_scale_code_r_single"
    assert schedule is not kernel.get_callees()[0]

    cpp = LFRicKokkosTrans().apply(loop)
    assert "eval_scale_r_single_kokkos" in cpp
    assert "eval_scale_kokkos" not in cpp
    assert "Kokkos::View<float*" in cpp
    assert "Kokkos::View<const double***, Kokkos::LayoutLeft, MemorySpace, " \
        "ReadOnly> basis_w1_on_w3(basis_w1_on_w3_data, 3, ndf_w1, ndf_w3);" \
        in cpp
    assert "weights" not in cpp


def test_lfric_kokkos_trans_selects_a_member_on_an_any_space(
        any_space_evaluator_target):
    """A basis on an ``any_space`` does not stop the question being asked.

    Metadata fixes a basis's first extent for a named function space and
    cannot for an ``any_space``, which is PSyclone's issue #461. That extent
    is not one of the things the matcher compares -- it is a literal on both
    sides -- so failing to produce it must not be read as the members failing
    to match. Before this task it was: the interface builder raised, the
    refusal said "found no implementation matching the precisions", and eight
    GungHo loops were reported as having no matching member when their
    precisions had never been looked at.
    """
    _, _, kernel = any_space_evaluator_target
    assert len(kernel.get_callees()) == 2

    # pylint: disable-next=protected-access
    schedule = LFRicKokkosTrans._schedule(kernel)
    assert schedule.name == "any_scale_code_r_single"


def test_lfric_kokkos_trans_refuses_an_ambiguous_evaluator_interface(
        ambiguous_evaluator_target):
    """Two members of an evaluator interface matching equally is a refusal.

    The companion of
    :py:func:`test_lfric_kokkos_trans_refuses_an_ambiguous_interface`, asked
    through the metadata this task makes askable. Widening what the matcher
    can be asked about must not turn a refusal into a first-match, so the
    refusal is asserted to name both members rather than to merely happen.
    """
    _, loop, _ = ambiguous_evaluator_target
    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)
    assert "found 2 implementations of 'dual_eval_code'" in str(error.value)
    assert "dual_eval_code_r_single, dual_eval_code_r_solver" in str(
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
    # Fortran declares from 1 and C indexes from 0, as for a formal. The
    # offset is the declared origin rendered as C, not an assumed 1, so it is
    # the string the backend writes into the subscript.
    assert all(item.index_offsets == ("1",) for item in scratch)


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


def test_lfric_kokkos_trans_accepts_a_zero_based_local(
        zero_based_local_target):
    """``dimension(0:nlayers)`` is captured, sized and shifted by its origin.

    This is the shape the catalogue's ``array-bound`` row counts. The View is
    one element longer than the upper bound, because the extent is
    ``ub - lb + 1`` and not ``ub``; and every subscript of it is shifted by
    the declared origin rather than by the Fortran default of 1.

    The shift is emitted even though it is zero. ``u_e(k - 0)`` is the origin
    stated in the generated code, and a subscript that read ``u_e(k)`` would
    be indistinguishable from one the offset had never reached -- which is
    the failure this capability is at risk of, since it is a wrong answer
    rather than a refusal.
    """
    psy, loop, kernel = zero_based_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("u_e")

    assert LFRicKokkosTrans._extents(symbol) == ("(nlayers + 1)",)
    assert LFRicKokkosTrans._origins(symbol) == ("0",)
    assert LFRicKokkosTrans._extent_names(symbol) == {"nlayers"}

    cpp = LFRicKokkosTrans().apply(loop)

    assert "u_e_scratch_t::shmem_size((nlayers + 1))" in cpp
    assert "u_e_scratch_t u_e(team.team_scratch(0), (nlayers + 1));" in cpp
    # Every subscript of it, not only the one the assignment writes.
    assert "u_e((k - 0))" in cpp
    assert "u_e((nlayers - 0))" in cpp
    assert "u_e(((k + 1) - 0))" in cpp
    # A subscript that never met the offset would read exactly this.
    assert "u_e(k)" not in cpp
    # Scratch reaches no interface, as for any other kernel-local array.
    assert "u_e" not in str(psy.gen)


def test_lfric_kokkos_trans_zero_based_bounds_enquiries(
        zero_based_enquiry_target):
    """LBOUND, UBOUND and SIZE of a zero-based array move with its origin.

    Each is answered from the declaration, so each has to be answered from
    *both* declared bounds: ``LBOUND`` is the origin rather than the constant
    1, and ``SIZE`` is the extent rather than the upper bound. Getting either
    from the old assumption gives an answer that is wrong by one.
    """
    _, loop, _ = zero_based_enquiry_target

    cpp = LFRicKokkosTrans().apply(loop)

    # LBOUND(u_e, 1) is 0 and UBOUND(u_e, 1) is nlayers, from the
    # declaration; the loop the kernel wrote them into is the evidence.
    assert "for(k=(0 + 2); k<=nlayers; k+=1)" in cpp
    # SIZE(u_e, 1) is the extent, which is one more than the upper bound.
    assert "(u_e((k - 0)) + (nlayers + 1))" in cpp
    for name in ("LBOUND", "UBOUND", "SIZE", "lbound(", "ubound(", "size(u_e"):
        assert name not in cpp


def test_lfric_kokkos_trans_accepts_a_negative_lower_bound(
        negative_origin_local_target):
    """``dimension(-nlayers:nlayers)`` is sized and shifted symbolically.

    The origin is an expression rather than a literal, which is the case
    where the extent and the origin are genuinely different readings of the
    same declaration: a region that sized the View correctly and shifted its
    subscripts by 1 would index a View of the right size from the wrong
    place.

    The extent is rendered as the writer builds it, ``((nlayers -
    (-nlayers)) + 1)``, which is ``2 * nlayers + 1`` unsimplified; the
    back-end emits an extent verbatim rather than folding it, so what is
    asserted is what is compiled.
    """
    _, loop, kernel = negative_origin_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("u_e")

    assert LFRicKokkosTrans._extents(symbol) == (
        "((nlayers - (-nlayers)) + 1)",)
    assert LFRicKokkosTrans._origins(symbol) == ("(-nlayers)",)
    # The origin is named in the generated C++ too, so it has to be reachable
    # there as well as at the launch.
    assert LFRicKokkosTrans._extent_names(symbol) == {"nlayers"}

    cpp = LFRicKokkosTrans().apply(loop)

    assert ("u_e_scratch_t u_e(team.team_scratch(0), "
            "((nlayers - (-nlayers)) + 1));" in cpp)
    assert "u_e((k - (-nlayers)))" in cpp
    assert "u_e((nlayers - (-nlayers)))" in cpp


def test_lfric_kokkos_trans_rejects_an_unrenderable_extent(
        unrenderable_extent_target):
    """``dimension(nlayers**nlayers)`` is refused, naming the extent.

    The origin is the Fortran default here, so the refusal is the extent
    grammar's own. The two checks run in order and the first to fire hides
    the second, which is why the origin refusal is asserted over a separate
    declaration rather than over this one.

    The exponent is a name because a literal one is written as a product now
    and is accepted; ``pow`` reaches an extent only through an exponent whose
    value is not known where the launch is written.
    """
    _, loop, _ = unrenderable_extent_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "extents of 'swept'" in str(error.value)
    assert "found 'pow(nlayers, nlayers)'" in str(error.value)


def test_lfric_kokkos_trans_sizes_scratch_from_a_squared_extent(
        squared_extent_target):
    """``dimension(nlayers**2)`` sizes the scratch from a product.

    The extent grammar refuses a call because a `shmem_size` argument is the
    text the launch writes and nothing rewrites it. An integer power with a
    literal exponent is no longer a call, so this declaration crossed from
    the refusal above into the region without the grammar being touched.
    """
    _, loop, kernel = squared_extent_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert LFRicKokkosTrans._extents(
        schedule.symbol_table.lookup("swept")) == ("(nlayers * nlayers)",)

    cpp = LFRicKokkosTrans().apply(loop)

    assert "(nlayers * nlayers)" in cpp
    assert "pow(" not in cpp


def test_lfric_kokkos_trans_rejects_an_unrenderable_lower_bound(
        unrenderable_origin_target):
    """A declared origin that is not an integer expression is still refused.

    Widening the origin from "must be 1" to "any integer expression over
    named sizes" is not a widening to anything at all.
    ``dimension(nlayers**nlayers:nlayers)`` is written
    ``pow(nlayers, nlayers)``, which the grammar refuses in an origin exactly
    as it refuses it in an extent -- and here the consequence is a subscript
    shifted by a value the launch cannot evaluate rather than a wrongly sized
    View.
    """
    _, loop, _ = unrenderable_origin_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'swept'" in str(error.value)
    assert "declared origin" in str(error.value)
    assert "found 'pow(nlayers, nlayers)'" in str(error.value)


def test_lfric_kokkos_trans_moves_an_origin_and_an_extent_together(
        lower_bound_local_target):
    """``dimension(0:nlayers-1)`` moves the origin and computes the extent.

    Both answers come out of one reading of one declaration, and this is the
    shape where reading them apart would disagree: the upper bound is itself
    an expression, so an extent taken as ``ub`` and an origin taken as 1 are
    each wrong by one and in opposite directions.
    """
    _, loop, kernel = lower_bound_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("swept")

    assert LFRicKokkosTrans._extents(symbol) == ("((nlayers - 1) + 1)",)
    assert LFRicKokkosTrans._origins(symbol) == ("0",)

    cpp = LFRicKokkosTrans().apply(loop)

    assert ("swept_scratch_t swept(team.team_scratch(0), "
            "((nlayers - 1) + 1));" in cpp)
    assert "swept((k - 0))" in cpp


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


def test_lfric_kokkos_trans_places_a_local_sized_by_a_named_constant(
        named_constant_local_target):
    """``parameter :: nfaces = 4`` sizing a local is read from its value.

    This is the commonest kernel-local shape GungHo has, and before the
    extent was resolved it was refused: ``nfaces`` is not a kernel argument,
    so the launch was told it could not compute the size. It does not have
    to. The value is stated by the declaration standing beside the array, so
    it is substituted into the extent exactly as the same value is already
    substituted into the body wherever the kernel names it.

    The constant is a routine-local ``parameter``, which is the shape the
    kernels use, rather than one of the module's own; both reach the same
    resolution, and the module case is covered by
    ``test_lfric_kokkos_trans_folds_a_static_constant``.
    """
    psy, loop, kernel = named_constant_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("v_dot_n")

    assert LFRicKokkosTrans._extents(symbol) == ("4",)
    # Nothing is left for the launch to be asked for.
    assert LFRicKokkosTrans._extent_names(symbol) == set()

    cpp = LFRicKokkosTrans().apply(loop)

    assert "v_dot_n_scratch_t::shmem_size(4)" in cpp
    assert "v_dot_n_scratch_t v_dot_n(team.team_scratch(0), 4);" in cpp
    # The name is nowhere in the generated unit: not in the extent, not in
    # the body, and not across the interface.
    assert "nfaces" not in cpp
    assert "nfaces" not in str(psy.gen)


def test_lfric_kokkos_trans_places_a_local_sized_by_a_division(
        divided_local_target):
    """``dimension((nlayers + 1)/2)`` is carried, and the truncation is said.

    A quotient is the last shape the catalogue's ``local-array`` row counted
    that the extent grammar refused outright, and refusing it was
    conservative rather than correct: Fortran and C++ both truncate an
    integer quotient toward zero, so the extent computed at the launch is the
    extent the kernel declared.

    What neither language defines is an allocation of negative size, which a
    division can reach where subtraction is in the numerator. So the launch
    states the rule it is relying on in a comment and stops a negative extent
    rather than requesting it -- the guard is emitted because this extent
    divides, and no region without a division gains one.
    """
    _, loop, kernel = divided_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)
    symbol = schedule.symbol_table.lookup("u_local")

    assert LFRicKokkosTrans._extents(symbol) == ("((nlayers + 1) / 2)",)
    assert LFRicKokkosTrans._extent_names(symbol) == {"nlayers"}

    cpp = LFRicKokkosTrans().apply(loop)

    assert "u_local_scratch_t::shmem_size(((nlayers + 1) / 2))" in cpp
    assert ("u_local_scratch_t u_local(team.team_scratch(0), "
            "((nlayers + 1) / 2));" in cpp)
    # The guard is over the extent that divides, and it aborts rather than
    # allocating.
    assert "if ((((nlayers + 1) / 2)) < 0) {" in cpp
    assert "Kokkos::abort(" in cpp
    # The array that does not divide gains no guard of its own.
    assert "if ((nlayers) < 0)" not in cpp


def test_lfric_kokkos_trans_rejects_a_shapeless_local(shapeless_local_target):
    """A declaration with no shape is refused, saying whose statement its is.

    The rule reads declarations, and a deferred shape is not one: the message
    says which of the two it is, and points at the statement that carries the
    size instead. What has changed since it was written is that the statement
    is now read. The declaration is rewritten from the ALLOCATE before this
    rule is asked, so a kernel whose extents are values the region holds at
    entry reaches it as an ordinary automatic array and is captured; the rule
    still refuses the declaration it is asked about here, which is the one the
    kernel wrote rather than the one the conversion leaves behind.
    """
    _, loop, kernel = shapeless_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    with pytest.raises(TransformationError) as error:
        # pylint: disable-next=protected-access
        LFRicKokkosTrans._validate_locals(schedule)

    assert "'swept' to be declared with explicit bounds" in str(error.value)
    assert "a deferred shape" in str(error.value)
    assert "ALLOCATE in the kernel body" in str(error.value)

    # And the loop as a whole is now accepted, the ALLOCATE stating a size
    # the launch can compute where it reserves its scratch.
    LFRicKokkosTrans().validate(loop)


def test_lfric_kokkos_trans_reports_an_unwritable_shape(
        unwritable_shape_target):
    """A declared shape the C writer cannot render is refused, not raised.

    ``dimension(max(nlayers,1))`` is what ffsl_flux_z_nirvana_kernel_mod
    declares, and MAX has no C operator: the writer's own failure is a
    ``VisitorError``, which ``validate`` may not raise, so a caller asking
    whether the loop was capturable got an exception of the wrong type from
    inside the backend instead of an answer. The refusal names the array and
    carries the writer's reason.
    """
    _, loop, _ = unwritable_shape_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "declared shape of 'swept' as C" in str(error.value)


@pytest.mark.parametrize("fixture_name", [
    "literal_local_target", "arithmetic_local_target",
    "explicit_one_local_target", "lower_bound_local_target",
    "zero_based_local_target", "negative_origin_local_target",
    "named_constant_local_target", "divided_local_target",
    "allocate_local_target", "minval_target"])
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
    "unrenderable_origin_target", "unsized_expression_target",
    "unwritable_shape_target", "unknown_allocate_target",
    "looped_allocate_target", "option_allocate_target",
    "twice_allocate_target", "shapeless_allocate_target",
    "module_allocate_target", "tiny_target"])
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


def test_lfric_kokkos_trans_inherits_the_bounds_grammar_refusal(
        lower_bound_enquiry_target):
    """A bound of an array with an unwritable origin gets that message.

    ``_bounds`` already refuses that declaration, and the enquiry depends on
    the same bounds, so its refusal is passed through rather than paraphrased
    -- a reader gets the sentence that says which rule was broken.
    """
    _, loop, _ = lower_bound_enquiry_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires the declared origin of 'swept' to be an integer "
            "expression over named sizes, but found 'pow(nlayers, nlayers)'"
            in str(error.value))


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


def test_lfric_kokkos_trans_accepts_an_array_parameter(array_constant_target):
    """An array parameter is declared in the unit, not passed to it.

    Its values are in the Fortran and its module is private, so there is
    nothing for the PSy layer to import and no argument worth adding: the
    generated unit declares it itself, among the body's locals rather than at
    file scope so that a device compiler can read it. It is indexed as C
    indexes an array, and with the same origin removed that a View subscript
    has.
    """
    psy, loop, _ = array_constant_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const int face_order[4] = {1, 2, 3, 4};" in cpp
    assert "static const" not in cpp
    assert cpp.count("const int face_order") == 1
    assert "partial((face_order[(k - 1)] - 1))" in cpp
    assert "face_order[(1 - 1)]" in cpp
    assert "face_order" not in fortran


def test_lfric_kokkos_trans_refuses_a_reshaped_array_parameter(
        reshaped_constant_target):
    """An array parameter built by a call states no values to declare.

    ``reshape`` is evaluated by the compiler, not by PSyIR, so there is no
    element list to write into the generated unit -- and a rank above one
    would not be indexed in C by the subscripts the body writes in any case.
    """
    _, loop, _ = reshaped_constant_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'point': a module-level constant is written "
            "into the region as its value, and this one was not declared "
            "with a literal value" in str(error.value))


def test_lfric_kokkos_trans_accepts_a_folded_parameter_expression(
        folded_constant_target):
    """A parameter declared as an expression is carried as that expression.

    ``weight = 1.0_r_def / (n_moist + n_extra)`` names two other
    parameters, one of them another module's. None of the three names is
    anything the region could read, but between them they state a value it
    can carry, so each is replaced by what it was declared as and the
    arithmetic is left for the C++ compiler to do exactly as the Fortran
    compiler would have.
    """
    psy, loop, _ = folded_constant_target

    cpp = LFRicKokkosTrans().apply(loop)
    generated = str(psy.gen)

    assert "(1.0 / (3 + 2))" in cpp
    assert "weight" not in cpp
    assert "n_moist" not in generated


def test_lfric_kokkos_trans_accepts_a_module_scalar_variable(
        module_variable_target):
    """A public module scalar becomes a formal the PSy layer passes.

    ``profile_size`` is neither a constant nor an argument, so the region
    can neither carry its value nor find it already in the call. It is a
    name the PSy layer can ``use``, though, so the region takes it by value
    and the launch reads it where it is made: what the module holds at
    region entry is what the region sees.
    """
    psy, loop, _ = module_variable_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const int profile_size) {" in cpp
    assert "integer(c_int), value :: profile_size" in fortran
    assert ("use column_solve_kernel_mod, only : profile_heights, "
            "profile_size" in fortran)
    assert "loop0_stop, profile_heights, profile_size)" in fortran


def test_lfric_kokkos_trans_accepts_a_module_array_variable(
        module_variable_target):
    """A public module array of literal extents becomes a read-only View.

    An array is state rather than a value, so it crosses by reference as
    every read-only array formal does -- sized from the extents the
    declaration states, and indexed with the origin those extents give.
    """
    psy, loop, _ = module_variable_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const double *profile_heights_data" in cpp
    assert ("Kokkos::View<const double*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> profile_heights(profile_heights_data, 100);" in cpp)
    assert "profile_heights((profile_size - 1))" in cpp
    assert ("real(c_double), dimension(*), intent(in) :: profile_heights"
            in fortran)


def test_lfric_kokkos_trans_refuses_an_assigned_module_variable(
        assigned_module_variable_target):
    """A module variable the body writes is refused rather than dropped.

    The region is given the value the module holds when the launch is made
    and has no share in the module's storage, so an assignment inside it
    would be lost. Losing it silently would be a wrong answer rather than an
    unsupported kernel.
    """
    _, loop, _ = assigned_module_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'profile_size' from 'column_solve_kernel_mod': "
            "the body assigns to it" in str(error.value))


def test_lfric_kokkos_trans_rejects_an_allocatable_module_variable(
        allocatable_module_variable_target):
    """An allocatable module array states no shape to give the region.

    Its extents are set at run time by whatever allocated it, so there is
    nothing for the region to size a View from and nothing the generated
    interface could declare. A deferred shape is a shape with no bounds, so
    it is refused by the reading of the bounds rather than by a check of its
    own.
    """
    _, loop, _ = allocatable_module_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("requires 'profile_heights' to be declared with explicit bounds"
            in str(error.value))


def test_lfric_kokkos_trans_rejects_a_module_array_of_named_extent(
        sized_module_variable_target):
    """A module array sized by a name is refused, not sized by guess.

    The region builds its own View over the pointer it is handed, so the
    extent has to be something it can evaluate. ``n_profile`` is a module
    variable rather than one of the region's arguments, and reading it as
    anything would be inventing a size.
    """
    _, loop, _ = sized_module_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'profile_heights': the region would have to "
            "size it from n_profile" in str(error.value))


def test_lfric_kokkos_trans_refuses_a_routine_local_static(
        local_static_target):
    """A local with an initialiser is a static of the routine, not the module.

    ``integer(i_def) :: visits = 0`` is rtheta_bd_kernel_mod's ``upwind``
    shape: Fortran gives it the SAVE attribute, so it keeps its value from
    one call to the next and nothing outside the routine declares it. The
    PSy layer has no name to pass and the region has nowhere to keep it.
    """
    _, loop, _ = local_static_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'visits': it is declared in the routine with an "
            "initialiser, which Fortran gives the SAVE attribute"
            in str(error.value))


def test_lfric_kokkos_trans_refuses_a_logical_array_parameter(
        logical_array_constant_target):
    """An array parameter with no C type is refused, values or no values.

    ``[.true., .false., .true., .false.]`` states its elements perfectly
    well. What it has no answer for is the type of the declaration they
    would go into: a logical is on the ABI by conversion, which is per
    value, and an array of them is no more declarable in the generated body
    than it is passable.
    """
    _, loop, _ = logical_array_constant_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot carry 'sweep': only" in str(error.value))
    assert "arrays have a place in the generated translation unit" in str(
        error.value)


def test_lfric_kokkos_trans_reads_no_value_it_cannot_resolve(monkeypatch):
    """A constant whose module cannot be read contributes no value.

    ``_constant_value`` answers ``None`` rather than raising, because the
    caller that needs the symbol on the ABI is the one that reports the
    missing module by name -- and it reports it once, rather than every
    reference to it reporting it again.
    """
    symbol = DataSymbol(
        "eps", ScalarType(ScalarType.Intrinsic.REAL,
                          ScalarType.Precision.UNDEFINED),
        interface=ImportInterface(ContainerSymbol("nowhere_mod")))
    monkeypatch.setattr(
        DataSymbol, "resolve_type",
        lambda self: (_ for _ in ()).throw(SymbolError("no such module")))

    assert LFRicKokkosTrans._constant_value(symbol) is None


@pytest.mark.parametrize("declaration", [
    "REAL(KIND = r_def), DIMENSION(3), PUBLIC :: coefficients",
    "TYPE(field_type), PUBLIC :: state",
])
def test_lfric_kokkos_trans_reads_no_c_type_from_a_shape_or_a_type(
        declaration):
    """Only a scalar of an intrinsic type is read out of a declaration.

    The declaration text is the last resort for a symbol PSyIR could not
    model, and it is read for a width to pass a *value* at. A declaration
    carrying a shape is not one value, and one naming a derived type is not
    a width, so both are answered ``None`` rather than parsed further.
    """
    symbol = DataSymbol(
        declaration.split("::")[1].strip(),
        UnsupportedFortranType(declaration))

    assert LFRicKokkosTrans._declared_c_type(symbol) is None


def _integer_literal(value):
    """Return one integer literal for the folding tests.

    :param int value: the value the literal states.

    :returns: the literal.
    :rtype: :py:class:`psyclone.psyir.nodes.Literal`
    """
    return Literal(str(value), ScalarType(
        ScalarType.Intrinsic.INTEGER, ScalarType.Precision.UNDEFINED))


def test_lfric_kokkos_trans_folds_a_reference_to_a_constant():
    """Folding a bare reference replaces the whole expression.

    Everything else is replaced inside the copy being folded, but an
    expression that *is* a reference has no parent to be replaced in, so the
    value becomes the result rather than being written into it.
    """
    symbol = DataSymbol(
        "nfaces", ScalarType(ScalarType.Intrinsic.INTEGER,
                             ScalarType.Precision.UNDEFINED),
        is_constant=True, initial_value=_integer_literal(4),
        interface=StaticInterface())

    folded = LFRicKokkosTrans._fold(Reference(symbol))

    assert isinstance(folded, Literal)
    assert folded.value == "4"


def test_lfric_kokkos_trans_folds_nothing_that_is_not_arithmetic():
    """An expression holding a node the region could not carry is not folded.

    A subscript-free substitution can put arithmetic over names and literals
    into the body and have it mean the same thing. It cannot do that for a
    call: what the compiler would have evaluated is not there to evaluate.
    """
    expression = IntrinsicCall.create(
        IntrinsicCall.Intrinsic.ABS, [_integer_literal(4)])

    assert LFRicKokkosTrans._fold(expression) is None


def test_lfric_kokkos_trans_folds_nothing_that_names_a_variable():
    """A name with no declared value stops the fold rather than surviving it.

    The region has no ``count`` to read, so an expression naming one states
    no value however much arithmetic surrounds it.
    """
    symbol = DataSymbol(
        "count", ScalarType(ScalarType.Intrinsic.INTEGER,
                            ScalarType.Precision.UNDEFINED),
        interface=StaticInterface())

    assert LFRicKokkosTrans._fold(Reference(symbol)) is None


def test_lfric_kokkos_trans_reads_no_array_from_an_unresolvable_import():
    """A constant whose module cannot be read contributes no value.

    ``_constant_value`` answers ``None`` rather than raising, because the
    caller that needs the symbol on the ABI is the one that reports the
    missing module by name.
    """
    symbol = DataSymbol(
        "eps", ScalarType(ScalarType.Intrinsic.REAL,
                          ScalarType.Precision.UNDEFINED),
        interface=ImportInterface(ContainerSymbol("nowhere_mod")))

    assert LFRicKokkosTrans._constant_value(symbol) is None


def _array_parameter(initial_value):
    """Return one ``parameter`` array symbol with the value given.

    :param initial_value: the initialiser the declaration carries.
    :type initial_value: :py:class:`psyclone.psyir.nodes.DataNode`

    :returns: the symbol.
    :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
    """
    integer = ScalarType(
        ScalarType.Intrinsic.INTEGER, ScalarType.Precision.UNDEFINED)
    return DataSymbol(
        "face_order", ArrayType(integer, [4]), is_constant=True,
        initial_value=initial_value, interface=StaticInterface())


def _absolute_four():
    """Return ``ABS(4)``, a value the region cannot carry.

    :returns: the call.
    :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
    """
    return IntrinsicCall.create(
        IntrinsicCall.Intrinsic.ABS, [_integer_literal(4)])


@pytest.mark.parametrize("symbol", [
    Symbol("face_order"),
    _array_parameter(_absolute_four()),
    _array_parameter(ArrayConstructor.create([_absolute_four()])),
])
def test_lfric_kokkos_trans_declares_no_array_it_cannot_read(symbol):
    """Only an array whose every element states a value is declared.

    A symbol PSyIR never specialised has no declaration to read at all; one
    initialised by a call rather than by a constructor has no element list
    to read; and one whose constructor holds a call has an element that
    states no value. None of the three has values to write into the
    generated unit, and each is answered ``None`` rather than
    half-declared.
    """
    assert LFRicKokkosTrans._constant_array(symbol) is None


def test_lfric_kokkos_trans_refuses_a_module_logical_array(
        logical_module_variable_target):
    """A module array off the ABI is refused as an argument would be.

    A logical crosses by conversion, which is per value; an array crosses by
    reference, and a ``View<bool*>`` over ``logical(l_def)`` storage would
    reinterpret its elements rather than convert them. Coming from a module
    rather than from the argument list changes none of that.
    """
    _, loop, _ = logical_module_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'profile_active' from 'column_solve_kernel_mod'"
            in str(error.value))
    assert "arrays of them have a place on the generated C ABI" in str(
        error.value)


def test_lfric_kokkos_trans_refuses_a_wildcard_imported_name(
        wildcard_import_target):
    """A name from a wildcard ``use`` is not an import the region can repeat.

    ``use planet_config_mod`` states no names, so the symbol carries no
    container to resolve a kind in and no declaration to read a shape from.
    It is not the kernel's own module's either, so there is nothing to
    describe and the refusal says so.
    """
    _, loop, _ = wildcard_import_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'recip_epsilon': it is neither a kernel argument "
            "nor imported from a module" in str(error.value))


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


def test_lfric_kokkos_trans_accepts_a_protected_logical_constant(
        off_abi_constant_target):
    """An imported logical constant crosses by the same conversion.

    ``rehabilitate`` is declared as gungho's configuration modules declare
    every namelist logical: ``logical(l_def), public, protected``, its kind
    stated positionally and its value assigned by the namelist reader rather
    than by a ``parameter``. PSyIR models none of that, so the type is read
    back out of the declaration text.

    It reaches the region as an argument rather than a literal, because its
    value is only known where the PSy layer runs, so the same wrapping applies
    to it as to a formal -- which is the point of doing the wrapping over
    ``region.arguments`` rather than over the formals alone.
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
    logical clause and the unkinded one.
    """
    _, loop, _ = unmapped_constant_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot pass 'unmapped_width' from 'planet_config_mod' by value"
            in str(error.value))
    assert "4-byte integer" in str(error.value)
    assert "logical of any kind" in str(error.value)
    assert "integer or logical declared with no kind" in str(error.value)


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
    assert "cannot inline the call to 'helper'" in str(error.value)


def test_lfric_kokkos_trans_inlines_a_module_procedure(
        module_procedure_target):
    """A procedure of the kernel's own module is inlined into the region.

    The generated region has no Fortran to call into, so the callee's
    statements have to become the kernel's own before the backend sees them.
    ``InlineTrans`` binds the formals to the actuals as it goes, so the
    inlined body reads ``partial`` and ``swept`` where the callee wrote
    ``source`` and ``result``.
    """
    _, loop, kernel = module_procedure_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    # The callee's own statements, with its formals bound to the actuals.
    assert "swept((nlayers - 1)) = partial((nlayers - 1));" in cpp


def test_lfric_kokkos_trans_inlines_a_call_with_a_section_actual(
        section_actual_target):
    """A contiguous section given as an actual reaches the inlined body.

    ``column(:,1)`` is a section outside an assignment, which is beyond what
    the section lowering can reach and is left to this rule by
    :py:meth:`_validate_sections`. Inlining removes the argument altogether:
    every use of the formal becomes a subscript of the local the section was
    taken from, so the region never has to pass an array at all.
    """
    _, loop, kernel = section_actual_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    assert "column" in cpp


def test_lfric_kokkos_trans_inlines_a_chain(chained_procedure_target):
    """A callee that itself calls is inlined to a fixed point.

    Inlining one call exposes the next, so the rewrite is repeated rather than
    run once. The repetition is bounded by
    :py:attr:`LFRicKokkosTrans._INLINE_LIMIT`, which is what makes a chain
    terminate on its own terms rather than on the recursion limit.
    """
    _, loop, kernel = chained_procedure_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    assert "seed_column" not in cpp
    assert LFRicKokkosTrans._INLINE_LIMIT > 1


def test_lfric_kokkos_trans_refuses_an_unresolvable_callee(
        called_routine_target):
    """A callee whose module was never read is refused, naming both.

    The message keeps PSyclone's own text, which names the container the
    symbol was imported from, so a reader is told which module to put on the
    search path rather than that something unnamed could not be found.
    """
    _, loop, _ = called_routine_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'helper'" in message
    assert "helper_mod" in message


def test_lfric_kokkos_trans_refuses_a_recursive_callee(
        recursive_procedure_target):
    """A callee that calls itself is refused rather than inlined forever.

    ``InlineTrans`` has no recursion check of its own -- inlining the body
    simply leaves another call to the same routine behind -- so the depth
    limit is the whole of what makes this terminate.
    """
    _, loop, _ = recursive_procedure_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "sweep_column" in message
    assert str(LFRicKokkosTrans._INLINE_LIMIT) in message


def test_lfric_kokkos_trans_refuses_an_external_callee(
        external_callee_target):
    """A callee outside the kernel's module is refused rather than guessed at.

    ``helper_state_mod`` is on the search path here, so the callee resolves
    and the refusal is not about finding it. The callee reads a datum its own
    module keeps private, so its body cannot be moved into the kernel's
    Container and cannot be inlined from where it is; PSyclone's own message
    says which Container the call site is in.
    """
    _, loop, _ = external_callee_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    assert "column_solve_kernel_mod" in message


def test_lfric_kokkos_trans_refuses_an_array_like_call(
        array_like_call_target):
    """A name PSyclone reads as a call and cannot type is refused, not raised.

    ``blend_weights(1)`` is an element of an array ``weights_config_mod``
    keeps, but the kernel's own file does not say so and the frontend leaves
    it as a Call. Asking PSyclone for that callee reaches the datum and
    raises :py:exc:`TypeError` rather than a
    :py:class:`~psyclone.psyir.transformations.TransformationError`, since a
    DataSymbol cannot be specialised into a RoutineSymbol. The mixin turns
    that into a refusal, so the caller is told the kernel is not captured
    instead of seeing PSyclone's traceback.
    """
    _, loop, _ = array_like_call_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'blend_weights'" in message
    assert "specialise" in message


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


def test_lfric_kokkos_trans_refuses_a_private_module_variable(
        static_variable_target):
    """A module variable the module keeps to itself cannot be passed.

    ``real(kind=r_def) :: cached_tol = 1.0e-9_r_def`` is state rather than a
    constant however it was initialised, so the region has to be given its
    value rather than write the initialisation in. A kernel module is
    ``private`` by default, though, and this one does not name ``cached_tol``
    in its ``public`` list: there is no name for the PSy layer to import.
    """
    _, loop, _ = static_variable_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert ("cannot capture 'cached_tol' from 'column_solve_kernel_mod': the "
            "module declares it private" in str(error.value))


def test_lfric_kokkos_trans_passes_a_repeated_import_once(
        repeated_import_target):
    """An imported constant read twice reaches the ABI once.

    Each reference is a separate node, so the walk meets ``recip_epsilon``
    twice; describing it twice would re-resolve a symbol already on the ABI
    and hand the PSy layer the same argument in two places.
    """
    psy, loop, kernel = repeated_import_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert [description[:4]
            for description in LFRicKokkosTrans._constants(schedule)] == [
        ("recip_epsilon", "planet_config_mod", None, "double")]

    LFRicKokkosTrans().apply(loop)

    generated = str(psy.gen)
    assert generated.count("map_wtheta, loop0_stop, recip_epsilon)") == 1


def test_lfric_kokkos_trans_carries_a_renamed_import(renamed_import_target):
    """A constant renamed on import is typed, and the rename is repeated.

    ``use planet_config_mod, only : recip => recip_epsilon`` leaves the body
    reading ``recip``, a name ``planet_config_mod`` does not declare. Typing
    it means looking it up in the module under the name it has there, and
    importing it into the PSy layer means writing the rename out again: a
    plain ``use planet_config_mod, only : recip`` would not compile.
    """
    psy, loop, kernel = renamed_import_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    assert [description[:4]
            for description in LFRicKokkosTrans._constants(schedule)] == [
        ("recip", "planet_config_mod", "recip_epsilon", "double")]

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert "const double recip" in cpp
    assert "use planet_config_mod, only : recip=>recip_epsilon" in fortran
    assert "map_wtheta, loop0_stop, recip)" in fortran


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


def test_lfric_kokkos_trans_iteration_space_predicate_refuses_a_dof_loop(
        target):
    """The iteration-space rule is askable on its own.

    The coverage survey reports every blocker a loop carries rather than the
    first, so each rule has to be a predicate of its own with the message the
    bundled check used to give.
    """
    _, loop, _ = target
    loop._iteration_space = "dof"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_iteration_space(loop)

    assert ("LFRicKokkosTrans supports only an uncoloured cell-column loop."
            in str(error.value))


def test_lfric_kokkos_trans_halo_depth_predicate_refuses_a_depth(target):
    """A halo depth is one of the two rules about where the loop runs."""
    _, loop, _ = target
    loop._upper_bound_halo_depth = Literal(
        "1", ScalarType(ScalarType.Intrinsic.INTEGER,
                        ScalarType.Precision.UNDEFINED))

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_halo_depth(loop)

    assert ("LFRicKokkosTrans does not support a halo depth."
            in str(error.value))


def test_lfric_kokkos_trans_halo_depth_predicate_refuses_halo_bounds(target):
    """A depthless halo bound is refused by the same predicate.

    'cell_halo' with no depth carries no halo depth to find, so the bounds
    rule is what catches it. The two belong together: both are about which
    cells the loop visits, and a survey reporting them apart would count one
    fact twice.
    """
    _, loop, _ = target
    loop._upper_bound_name = "cell_halo"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_halo_depth(loop)

    assert ("LFRicKokkosTrans supports only owned-cell bounds."
            in str(error.value))


def test_lfric_kokkos_trans_evaluator_predicate_refuses_an_unmodelled_shape(
        target):
    """The evaluator-shape rule is askable on its own, shape by shape."""
    _, _, kernel = target
    kernel._basis_required = True
    kernel._eval_shapes = ["gh_quadrature_edge"]

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_evaluator(kernel)

    assert ("LFRicKokkosTrans does not support the 'gh_quadrature_edge' "
            "evaluator shape." in str(error.value))


def test_lfric_kokkos_trans_evaluator_predicate_accepts_modelled_shapes(
        target):
    """The two shapes the region models pass the rule, together or apart.

    Asked of each shape separately and of both at once, because a rule
    written to accept a single shape would refuse the kernels that ask for
    both -- and those are the ones the model has most of.
    """
    _, _, kernel = target
    for shapes in (["gh_quadrature_xyoz"], ["gh_evaluator"],
                   ["gh_quadrature_xyoz", "gh_evaluator"]):
        kernel._eval_shapes = shapes
        LFRicKokkosTrans._validate_evaluator(kernel)


def test_lfric_kokkos_trans_field_type_predicate_refuses_a_logical_field(
        target):
    """The field-type rule walks every field argument on its own.

    LFRic's metadata admits ``gh_real`` and ``gh_integer`` fields and no
    third intrinsic, and the ABI now carries both, so the witness has to be
    made rather than found. Made rather than deleted, because the rule is
    stated against what a View's elements may be and not against today's
    metadata: a field intrinsic added to LFRic would be refused by name
    instead of reaching the backend as a type it has no row for.
    """
    _, _, kernel = target
    kernel.arguments.args[1]._intrinsic_type = "logical"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_field_types(kernel)

    assert ("LFRicKokkosTrans supports only integer and real fields, but "
            "'mr' is logical." in str(error.value))


def test_lfric_kokkos_trans_continuous_write_predicate_refuses_a_w0_write(
        target):
    """The written-space rule walks every field argument on its own."""
    _, _, kernel = target
    kernel.arguments.args[0].function_space._orig_name = "w0"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_continuous_write(kernel)

    assert ("LFRicKokkosTrans requires a discontinuous space for the written "
            "field 'moist_dyn', but found 'w0': one cell's contribution "
            "could overwrite another's." in str(error.value))


def test_lfric_kokkos_trans_predicates_accept_a_supported_loop(target):
    """Each of the five returns for a loop that does not fail it.

    A predicate that raised for everything would report every pattern as
    blocked, which is the failure mode a survey cannot see from the inside.
    """
    _, loop, kernel = target

    LFRicKokkosTrans._validate_iteration_space(loop)
    LFRicKokkosTrans._validate_halo_depth(loop)
    LFRicKokkosTrans._validate_evaluator(kernel)
    LFRicKokkosTrans._validate_field_types(kernel)
    LFRicKokkosTrans._validate_continuous_write(kernel)


def test_lfric_kokkos_trans_continuous_write_predicate_ignores_the_shape(
        target):
    """A predicate answers for its own rule, not for the first blocker.

    This is the whole point of naming the five. A kernel that asks for an
    unmodelled evaluator shape *and* writes a continuous space is two blocked
    patterns, and asking through 'validate' would only ever name the shape
    because it is checked first.
    """
    _, _, kernel = target
    kernel._eval_shapes = ["gh_quadrature_face"]
    kernel.arguments.args[0].function_space._orig_name = "w0"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_continuous_write(kernel)
    assert "discontinuous space for the written field 'moist_dyn'" in str(
        error.value)

    with pytest.raises(TransformationError) as second:
        LFRicKokkosTrans._validate_evaluator(kernel)
    assert "does not support the 'gh_quadrature_face' evaluator shape" in str(
        second.value)


def test_lfric_kokkos_trans_metadata_refuses_by_argument_order(target):
    """The bundled check still refuses in argument order, not rule order.

    The first argument is written to a continuous space and the second is a
    non-real field. Walking the arguments -- which is what the transformation
    has always done -- reports the first argument's blocker; running the two
    rules as separate passes over all the arguments would report the second
    argument's instead. The predicates share the per-argument helpers with
    this loop so that only one answer exists.
    """
    _, _, kernel = target
    kernel.arguments.args[0].function_space._orig_name = "w0"
    kernel.arguments.args[1]._intrinsic_type = "integer"

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans._validate_kernel_metadata(kernel)

    assert "discontinuous space for the written field 'moist_dyn'" in str(
        error.value)
    assert "real fields" not in str(error.value)


def test_lfric_kokkos_trans_field_predicates_pass_over_a_non_field(
        operator_target):
    """The two field rules walk the whole argument list and skip the rest.

    Asked through 'validate' they only ever see an argument the walk has
    already accepted as a field. Asked on their own -- which is how the
    survey asks them -- they meet the scalars and operators too, and a rule
    that read a function space off an LMA operator would raise something
    other than a refusal.
    """
    _, _, kernel = operator_target
    assert any(argument.argument_type != "gh_field"
               for argument in kernel.arguments.args)

    LFRicKokkosTrans._validate_field_types(kernel)
    LFRicKokkosTrans._validate_continuous_write(kernel)


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


def test_kokkos_allocate_becomes_scratch(allocate_local_target):
    """An ALLOCATE whose extents are region-entry values becomes scratch.

    A kernel-local ``allocatable`` sized from the kernel's own arguments and
    freed before the routine returns is the automatic array the local-array
    branch already places in team scratch, written the other way round. The
    conversion rewrites the declaration from the ALLOCATE and removes both
    statements, after which every rule about a local array applies unchanged.
    """
    _, loop, _ = allocate_local_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "team.team_scratch" in cpp
    assert "partial" in cpp and "swept" in cpp
    for name in ("allocate", "ALLOCATE", "deallocate", "DEALLOCATE"):
        assert name not in cpp


def test_kokkos_allocate_rewrites_the_declaration(allocate_local_target):
    """The converted local is described exactly as a declared one is."""
    _, loop, kernel = allocate_local_target
    schedule = LFRicKokkosTrans._schedule(kernel)

    LFRicKokkosTrans._lower_allocations(schedule)

    assert [(scratch.name, scratch.extents, scratch.index_offsets)
            for scratch in LFRicKokkosTrans._local_arrays(schedule)] \
        == [("partial", ("nlayers",), ("1",)),
            ("swept", ("nlayers",), ("1",))]
    assert not [call for call in schedule.walk(IntrinsicCall)
                if call.intrinsic in (IntrinsicCall.Intrinsic.ALLOCATE,
                                      IntrinsicCall.Intrinsic.DEALLOCATE)]
    assert loop is not None


def test_kokkos_allocate_refused_when_the_extent_is_not_known(
        unknown_allocate_target):
    """An extent the launch cannot evaluate is refused, with the array named.

    The scratch size is computed on the host before the launch, where the
    only values in scope are the region's own scalars. A reduction over a
    kernel argument is not one of them, so this is a refusal rather than a
    conversion.
    """
    _, loop, _ = unknown_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'partial'" in str(error.value)


def test_kokkos_allocate_refused_inside_a_loop(looped_allocate_target):
    """An allocation inside a loop is a different array each trip."""
    _, loop, _ = looped_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'partial'" in str(error.value)
    assert "outside every loop" in str(error.value)


def test_lfric_kokkos_trans_generates_minval(minval_target):
    """MINVAL over a kernel-local generates, closing wave A's skip.

    ``conservative_neg_fix_code`` was accepted by ``validate`` and refused by
    the writer, and was skipped in the optimisation script for that reason.
    The reduction is now written, so the skip has nothing left to protect.
    """
    _, loop, _ = minval_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "Kokkos::reduction_identity<double>::min()" in cpp
    assert "Kokkos::min(" in cpp
    assert "MINVAL" not in cpp


def test_lfric_kokkos_trans_refuses_an_intrinsic_the_writer_lacks(
        tiny_target):
    """An intrinsic no writer can spell is refused by validate, by name.

    ``validate`` used to accept any body whose *shape* was capturable and
    leave the writer to discover that it could not spell one of its
    intrinsics, which surfaced as a back-end error out of ``apply``. The
    check asks the writer's own tables, so the two cannot part company.
    """
    _, loop, _ = tiny_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "TINY/1" in str(error.value)


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


def test_kokkos_allocate_refused_when_it_carries_an_option(
        option_allocate_target):
    """An option beside the arrays says something scratch cannot say.

    ``stat=``, ``errmsg=``, ``source=`` and ``mold=`` each state part of what
    the allocation is to do, and the reserved scratch carries none of it.
    Dropping the statement would drop the option with it, so the refusal
    names it, and names the arguments as they were written.
    """
    _, loop, _ = option_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'mold'" in str(error.value)
    assert "'0.0_r_def'" in str(error.value)


def test_kokkos_allocate_refused_when_allocated_twice(twice_allocate_target):
    """Two allocations of one local are two shapes over one reservation."""
    _, loop, _ = twice_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'partial'" in str(error.value)
    assert "allocated more than once" in str(error.value)


def test_kokkos_allocate_refused_when_it_states_no_shape(
        shapeless_allocate_target):
    """An allocation stating no bounds leaves nothing to reserve."""
    _, loop, _ = shapeless_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'partial'" in str(error.value)
    assert "states no explicit shape" in str(error.value)


def test_kokkos_allocate_refused_for_a_module_array(module_allocate_target):
    """Allocating module storage is not the kernel-local case.

    A module-scope allocatable outlives the region and is shared with
    whatever else the module lets at it, so giving it the shape the statement
    states and reserving scratch for it would move storage the model reads
    after the invoke.
    """
    _, loop, _ = module_allocate_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'profile_heights'" in str(error.value)
    assert "not a kernel-local allocatable array" in str(error.value)


# set_exner_code's shape: a whole array assigned the value of a MATMUL,
# `exner_e(:) = MATMUL(inv_mass_matrix_w3, rhs_e)`. The section lowering
# refuses this assignment -- ArrayAssignment2LoopsTrans takes only a
# scalar-valued or elemental right-hand side -- so it has to be left to the
# backend, which writes the contraction as a nest over the destination.
_MATMUL_KERNEL = _LOCAL_KERNEL.replace(
    "    integer(kind=i_def) :: k\n",
    "    integer(kind=i_def) :: k\n"
    "    real(kind=r_def), dimension(3,3) :: mass\n"
    "    real(kind=r_def), dimension(3) :: rhs_e\n"
    "    real(kind=r_def), dimension(3) :: column_e\n").replace(
    "    swept(nlayers) = partial(nlayers)",
    "    mass(:,:) = 1.0_r_def\n"
    "    rhs_e(:) = partial(1)\n"
    "    column_e(:) = matmul(mass, rhs_e)\n"
    "    swept(nlayers) = partial(nlayers) + column_e(3)")


@pytest.fixture(name="matmul_target")
# pylint: disable-next=unused-argument
def matmul_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose kernel assigns a whole array from MATMUL."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _MATMUL_KERNEL)


def test_lfric_kokkos_trans_generates_a_matmul_assignment(matmul_target):
    """A whole array assigned from MATMUL is left to the backend.

    The section lowering is what rewrites `a(:) = ...` into a loop, and it
    declines a right-hand side that is neither scalar-valued nor elemental.
    Refusing on that would refuse the very shape this tier exists for, so an
    assignment holding one of its intrinsics is kept from the lowering and
    written as a nest instead.
    """
    _, loop, _ = matmul_target

    cpp = LFRicKokkosTrans().apply(loop)

    assert "MATMUL" not in cpp
    assert "+= mass(" in cpp
    assert "column_e((_kae_i0 - 1)) = _kae_r0;" in cpp


def test_lfric_kokkos_trans_accepts_an_integer_field(
        integer_field_target, real_field_target):
    """A field whose data is integer crosses the ABI as a View of int.

    The element type follows the argument, and nothing else about the field
    does: the dofmap, the ndf and undf formals and the launch belong to the
    function space rather than to the intrinsic. They are asserted to be the
    real kernel's to the character, which the two regions being generated
    from one template makes a statement rather than a coincidence.
    """
    psy, loop, _ = integer_field_target
    _, real_loop, _ = real_field_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)
    real_cpp = LFRicKokkosTrans().apply(real_loop)

    assert "int *mask_out_data," in cpp
    assert "const int *mask_in_data," in cpp
    assert ("Kokkos::View<int*, Kokkos::LayoutLeft, MemorySpace, Unmanaged> "
            "mask_out(mask_out_data, undf_w3);" in cpp)
    assert ("Kokkos::View<const int*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> mask_in(mask_in_data, undf_wtheta);" in cpp)
    # The dofmap and the sizes are what they are for a real field.
    assert "const int undf_w3," in cpp
    assert "const int undf_wtheta," in cpp
    assert ("Kokkos::View<const int**, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> map_w3(map_w3_data, ndf_w3, ncells);" in cpp)
    assert "Kokkos::RangePolicy<>(0, ncells)" in cpp

    # The width the C++ body assumed for i_def, asserted where the generated
    # Fortran and the generated C++ meet. A region carrying integer data and
    # nothing else names no real kind at all, so this is the only assertion
    # it writes and c_double never reaches its 'use' line.
    assert "use iso_c_binding, only : c_int" in fortran
    assert "use constants_mod, only : i_def" in fortran
    assert ("storage_size(1_i_def) == &\n        storage_size(1_c_int))), "
            "parameter :: assert_kind_i_def = 0" in fortran)
    assert "c_double" not in fortran
    assert "integer(c_int), dimension(*), intent(inout) :: mask_out" in fortran
    assert "call int_ratio_kokkos(" in fortran
    assert "call mask_out_proxy%set_dirty()" in fortran

    # Everything the intrinsic does not decide is the real kernel's: putting
    # the element type and the kernel's name back gives that region exactly.
    assert real_cpp.replace("double", "int").replace(
        "real_ratio", "int_ratio") == cpp
    assert "double" not in cpp


def test_lfric_kokkos_trans_accepts_a_mixed_field_kernel(mixed_field_target):
    """Each field formal takes its own element type, in the order given.

    The intrinsic is per argument rather than per kernel, and a real field
    beside an integer one is what separates the two readings. The order is
    asserted as well as the types, because the generated signature and the
    actuals the PSy layer passes carry their correspondence in nothing but
    their shared indices: a formal given the wrong element type is a region
    that compiles and reads the wrong storage.
    """
    psy, loop, _ = mixed_field_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    assert ("    double *field_out_data,\n"
            "    const double *field_in_data,\n"
            "    const int *mask_data,\n" in cpp)
    assert ("Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace, "
            "Unmanaged> field_out(field_out_data, undf_w3);" in cpp)
    assert ("Kokkos::View<const int*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> mask(mask_data, undf_wtheta);" in cpp)

    # One interface carrying both widths, and both kinds asserted on it.
    assert "use iso_c_binding, only : c_int, c_double" in fortran
    assert "use constants_mod, only : i_def, r_def" in fortran
    assert ("storage_size(1_i_def) == &\n        storage_size(1_c_int))), "
            "parameter :: assert_kind_i_def = 0" in fortran)
    assert ("storage_size(1.0_r_def) == &\n        "
            "storage_size(1.0_c_double))), parameter :: assert_kind_r_def = 0"
            in fortran)
    assert ("real(c_double), dimension(*), intent(inout) :: field_out\n"
            "    real(c_double), dimension(*), intent(in) :: field_in\n"
            "    integer(c_int), dimension(*), intent(in) :: mask\n"
            in fortran)
    assert ("call masked_copy_kokkos(nlayers_out_field, out_field_data, "
            "in_field_data, mask_data, ndf_w3, undf_w3, map_w3, ndf_wtheta, "
            "undf_wtheta, map_wtheta, loop0_stop)" in fortran)


def test_lfric_kokkos_trans_integer_field_arithmetic_is_integer(
        integer_field_target, real_field_target):
    """A quotient of two integer field values is an integer quotient.

    C++ takes '/' from its operands, so this is a statement about the Views
    the region declares rather than about the expression: the writer emits
    the same text for both kernels and the element type is the whole of what
    makes one of them truncate. Worth asserting because a promotion anywhere
    on the way -- a cast written round a field read, a double View over
    integer storage -- would change the answer rather than fail to compile.
    """
    _, loop, _ = integer_field_target
    _, real_loop, _ = real_field_target

    cpp = LFRicKokkosTrans().apply(loop)
    real_cpp = LFRicKokkosTrans().apply(real_loop)

    quotient = ("(mask_in(((map_wtheta((df - 1), cell) + k) - 1)) / "
                "mask_div(((map_wtheta((df - 1), cell) + k) - 1)))")
    assert quotient in cpp
    assert quotient in real_cpp
    assert ("Kokkos::View<const int*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> mask_div(mask_div_data, undf_wtheta);" in cpp)
    assert ("Kokkos::View<const double*, Kokkos::LayoutLeft, MemorySpace, "
            "ReadOnly> mask_div(mask_div_data, undf_wtheta);" in real_cpp)
    # Nothing widens the operands on the way to the division.
    assert "double" not in cpp
    assert "float" not in cpp
    assert "static_cast" not in cpp


def test_lfric_kokkos_trans_refuses_a_field_kind_not_on_the_abi(
        off_abi_field_target):
    """Following the argument's intrinsic did not retire the width check.

    ``i_native`` is an LFRic integer kind that ``psyclone.cfg``'s precision
    map does not carry, so there is no width to put on the ABI. Writing
    ``int`` for it because the argument is an integer field would drop or
    invent bytes without saying so, so it is refused naming the field and the
    kind. Nothing here is monkeypatched: the kind really is absent.
    """
    _, loop, _ = off_abi_field_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    assert "'mask_out' has 'i_native'" in str(error.value)
    assert "4-byte integer, 4-byte real and 8-byte real" in str(error.value)


def test_lfric_kokkos_trans_validate_keeps_the_module_scope_chain(
        module_index_target):
    """A name the kernel *module* imports is still resolvable in the probe.

    ``validate()`` predicts the rewrite on a copy, and the copy has to keep
    the FileContainer the kernel was read from: a Routine copied on its own
    leaves every symbol the module ``use``d at module level out of the scope
    chain. The dependence analysis is where that is felt, because comparing
    two subscripts symbolically means looking each name in them up --
    ``moist_dyn_gas(map_wtheta(n_moist) + k)`` on both sides of one
    assignment, with ``n_moist`` imported by the module.

    The failure this covers is not a refusal: it is
    ``KeyError: "Could not find 'n_moist' in the Symbol Table."`` coming
    straight out of ``SymPyWriter``, which breaks ``validate()``'s contract
    that a rejection is a ``TransformationError``. Found on LFRic's inter-grid
    prolongation, where the constant is ``SWB``; reproduced here on a
    single-mesh kernel, because nothing about it is inter-grid.
    """
    _, loop, _ = module_index_target

    try:
        LFRicKokkosTrans().validate(loop)
    except TransformationError as err:
        pytest.fail(f"the capture contract refused the kernel: {err}")


def test_lfric_kokkos_trans_validate_probe_is_not_the_schedule(section_target):
    """``validate()`` leaves the kernel schedule exactly as it found it.

    That is the whole reason the probe is a copy, and it is worth asserting
    directly rather than through the generated text: the probe is lowered,
    has its bounds substituted and its allocations resolved, and every one of
    those would be a side effect of *validating* if it reached the schedule.
    The section kernel witnesses it, because the lowering it would undergo is
    visible in the tree -- an array section becomes a loop nest -- rather than
    only in a symbol's type.
    """
    _, loop, kernel = section_target
    schedule = kernel.get_callees()[0]
    before = schedule.debug_string()
    assert schedule.walk(Range)

    LFRicKokkosTrans().validate(loop)

    assert kernel.get_callees()[0] is schedule
    assert schedule.debug_string() == before
    assert schedule.walk(Range)
