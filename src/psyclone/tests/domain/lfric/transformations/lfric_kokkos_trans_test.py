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
from psyclone.parse import ModuleManager
from psyclone.parse.algorithm import parse
from psyclone.psyGen import PSyFactory
from psyclone.psyir.backend.kokkos import KokkosRegion, KokkosScalar
from psyclone.psyir.nodes import (
    ArrayReference, CodeBlock, IntrinsicCall, Literal, Range)
from psyclone.psyir.symbols import ScalarType
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


# The same shape with the scalar made logical, so that the refusal stage 2
# deliberately leaves in place has a test of its own. LFRic's l_def is
# kind(.false.), which measures 4 bytes, but PSyclone's precision map records
# l_def as 1. Admitting it would generate logical(c_bool) against a logical(4)
# actual, which does not compile, so the ABI widening stops at float. See
# stage 2 of psy-ir-aidev/docs/plans/2026-08-29-phase-3-coverage.md.
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
  use constants_mod, only : r_def
  implicit none
  real(kind=r_def), public, protected :: recip_epsilon = 1.0_r_def
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


@pytest.fixture(name="second_target")
# pylint: disable-next=unused-argument
def second_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an unrelated supported kernel in a minimal LFRic invoke."""
    return _invoke(
        tmp_path, "scaled_copy", _SECOND_ALGORITHM, _SECOND_KERNEL)


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
    # own lower bound as an offset from the assigned range's start.
    assert "for(idx=w3_idx; idx<=(w3_idx + nl); idx+=1)" in cpp
    assert "difference((idx - 1)) = " in cpp
    assert "mass_flux(((idx + ((b_idx + 1) - w3_idx)) - 1))" in cpp
    assert "mass_flux(((idx + (b_idx - w3_idx)) - 1))" in cpp
    # The counter the lowering introduced is declared inside the lambda. It
    # names no Fortran kind, so it reaches the C writer's own default rather
    # than a kind the region described -- which is the fallback in
    # KokkosWriter.gen_declaration, load-bearing rather than defensive.
    assert "int idx;" in cpp
    assert "Kokkos::RangePolicy<>(0, ncells)" in cpp

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


def test_lfric_kokkos_trans_rejects_stencil(target):
    """Stencil storage and halo requirements are not silently captured."""
    _, loop, kernel = target
    kernel.arguments.args[1].stencil = {"type": "xory1d"}
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


def test_lfric_kokkos_trans_still_refuses_a_logical_kind(logical_target):
    """A logical scalar stays refused after the ABI admits single precision.

    LFRic's l_def is kind(.false.) and measures 4 bytes; PSyclone's precision
    map records it as 1. Generating logical(c_bool) against a logical(4)
    actual would not compile, so the widening deliberately stops at float.
    """
    _, loop, _ = logical_target
    with pytest.raises(TransformationError, match="argument kinds") as err:
        LFRicKokkosTrans().validate(loop)
    assert "masked" in str(err.value)


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
