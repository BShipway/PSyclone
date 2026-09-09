# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosInterfaceMixin: the bind(C) interface."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.backend.kokkos import KokkosRegion, KokkosScalar
from psyclone.psyir.transformations import TransformationError


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
