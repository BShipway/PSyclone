# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for resolving a generic interface by the kind of an array argument."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LOCAL_ALGORITHM, _LOCAL_KERNEL, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import Call, IntrinsicCall
from psyclone.psyir.transformations import TransformationError


# LFRic's own kind constants, as a module the test writes out beside the
# kernel. The tests have no `constants_mod` of their own, and the rule under
# test is what happens when PSyclone *can* read the module a kind is named
# in -- which in the model it always can, `constants_mod` being on the
# survey's search path. The three values are the ones a default LFRic build
# has, and are what `psyclone.cfg`'s precision map records for these names.
_CONSTANTS_MODULE = """
module constants_mod
  implicit none
  integer, parameter :: i_def = 4
  integer, parameter :: r_def = 8
  integer, parameter :: r_single = 4
  integer, parameter :: r_double = 8
end module constants_mod
"""


# The Jacobian family's shape, reduced to one kernel. `coordinate_jacobian`
# is a generic interface of specifics named for the real kind of their array
# arguments -- `..._r_single` and `..._r_double` -- which is the only thing
# that differs between them. The actual states a third name again, `r_def`,
# and the call is legal Fortran because `r_def` and one of the two hold the
# same value.
#
# The two specifics' bodies differ in a literal so that which of them was
# inlined can be read off the generated C++.
_GENERIC_ARRAY_KIND_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, r_def, r_double, r_single").replace(
    "end type column_solve_kernel_type\ncontains\n",
    "end type column_solve_kernel_type\n"
    "  interface sweep_column\n"
    "    ! Named single-first, so that a callee chosen by order rather than\n"
    "    ! by kind would be the wrong one.\n"
    "    module procedure sweep_column_single, sweep_column_double\n"
    "  end interface sweep_column\n"
    "contains\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column_double(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_double), dimension(n), intent(in) :: source\n"
    "    real(kind=r_double), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - 3.0_r_double * source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column_double\n"
    "  subroutine sweep_column_single(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_single), dimension(n), intent(in) :: source\n"
    "    real(kind=r_single), dimension(n), intent(inout) :: result\n"
    "    integer(kind=i_def) :: j\n"
    "    result(n) = source(n)\n"
    "    do j = n - 1, 1, -1\n"
    "      result(j) = result(j + 1) - 7.0_r_single * source(j)\n"
    "    end do\n"
    "  end subroutine sweep_column_single\n"
    "end module column_solve_kernel_mod")


# The same interface called with a kind none of the three names can be
# resolved to a value: every one of them comes from `constants_mod`, which
# the tests do not provide and PSyclone therefore cannot read. Two names it
# cannot resolve may hold the same value or different ones, so neither
# specific is refused and neither is preferred.
_UNRESOLVABLE_KIND_KERNEL = _LOCAL_KERNEL.replace(
    "  use constants_mod, only : i_def, r_def",
    "  use constants_mod, only : i_def, r_def, r_solver, r_tran").replace(
    "end type column_solve_kernel_type\ncontains\n",
    "end type column_solve_kernel_type\n"
    "  interface sweep_column\n"
    "    module procedure sweep_column_solver, sweep_column_tran\n"
    "  end interface sweep_column\n"
    "contains\n").replace(
    "    swept(nlayers) = partial(nlayers)\n"
    "    do k = nlayers - 1, 1, -1\n"
    "      swept(k) = swept(k + 1) - partial(k)\n"
    "    end do\n",
    "    call sweep_column(nlayers, partial, swept)\n").replace(
    "end module column_solve_kernel_mod",
    "  subroutine sweep_column_solver(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_solver), dimension(n), intent(in) :: source\n"
    "    real(kind=r_solver), dimension(n), intent(inout) :: result\n"
    "    result(n) = source(n)\n"
    "  end subroutine sweep_column_solver\n"
    "  subroutine sweep_column_tran(n, source, result)\n"
    "    integer(kind=i_def), intent(in) :: n\n"
    "    real(kind=r_tran), dimension(n), intent(in) :: source\n"
    "    real(kind=r_tran), dimension(n), intent(inout) :: result\n"
    "    result(n) = source(n)\n"
    "  end subroutine sweep_column_tran\n"
    "end module column_solve_kernel_mod")


@pytest.fixture(name="array_kind_target")
# pylint: disable-next=unused-argument
def array_kind_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose generic call is settled by an array's kind."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _GENERIC_ARRAY_KIND_KERNEL,
                   extra={"constants_mod": _CONSTANTS_MODULE})


@pytest.fixture(name="unresolvable_kind_target")
# pylint: disable-next=unused-argument
def unresolvable_kind_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke whose generic call names kinds nothing resolves."""
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _UNRESOLVABLE_KIND_KERNEL)


def test_array_kind_chooses_the_specific(array_kind_target):
    """The specific whose array kind is the actual's is the one captured.

    Nothing but the kind distinguishes the two, and the interface names the
    wrong one first, so a capture that took the first candidate would put
    the narrow specific's body in the region. The two bodies differ in a
    literal so that which of them arrived can be read off the C++.
    """
    _, loop, kernel = array_kind_target

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    assert "3.0" in cpp
    assert "7.0" not in cpp


def test_unresolvable_array_kinds_are_not_guessed_at(
        unresolvable_kind_target):
    """Two kinds PSyclone cannot resolve do not decide between specifics.

    `r_solver`, `r_tran` and `r_def` all come from a module the tests do not
    provide. Two names are not evidence of two values, so both specifics
    match weakly and the call is refused as ambiguous rather than settled by
    the order the interface names them in.
    """
    _, loop, _ = unresolvable_kind_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "Ambiguous call to 'sweep_column'" in message
