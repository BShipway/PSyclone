# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for a callee that calls its own module through a generic interface."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import (
    _LOCAL_ALGORITHM, _SIBLING_CALLEE_KERNEL, _invoke)

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import Call, IntrinsicCall
from psyclone.psyir.transformations import TransformationError


# The Jacobian family's shape. `sci_coordinate_jacobian_mod` publishes
# `coordinate_jacobian` and `pointwise_coordinate_jacobian` as generic
# interfaces, and the specifics behind them call `jacobian_abr2XYZ` and
# `jacobian_stretched` -- generic interfaces of that same module -- rather
# than naming a specific. Both layers are generic here for the same reason:
# the call the kernel makes is settled by one rule and the call the specific
# makes has to be settled by the same one.
#
# Each interface names first the specific the call does *not* select, so that
# a resolution by the order the interface lists them in would put the wrong
# body in the region. The two `damping` specifics differ in a literal, which
# is how the C++ says which of them arrived.
_GENERIC_SIBLING_MODULE = """
module sweep_support_mod
  use constants_mod, only : i_def, r_def
  implicit none
  private
  public :: sweep_column
  interface sweep_column
    module procedure sweep_column_damped, sweep_column_plain
  end interface sweep_column
  interface damping
    module procedure damping_weighted, damping_level
  end interface damping
contains
  subroutine sweep_column_plain(n, source, result)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    integer(kind=i_def) :: j
    result(n) = source(n) * damping(n)
    do j = n - 1, 1, -1
      result(j) = result(j + 1) - source(j) * damping(j)
    end do
  end subroutine sweep_column_plain
  subroutine sweep_column_damped(n, source, result, weight)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    real(kind=r_def), intent(in) :: weight
    integer(kind=i_def) :: j
    result(n) = source(n) * damping(n, weight)
    do j = n - 1, 1, -1
      result(j) = result(j + 1) - source(j) * damping(j, weight)
    end do
  end subroutine sweep_column_damped
  function damping_level(level) result(factor)
    integer(kind=i_def), intent(in) :: level
    real(kind=r_def) :: factor
    factor = 1.0_r_def / real(level + 3, r_def)
  end function damping_level
  function damping_weighted(level, weight) result(factor)
    integer(kind=i_def), intent(in) :: level
    real(kind=r_def), intent(in) :: weight
    real(kind=r_def) :: factor
    factor = 9.0_r_def * weight / real(level + 3, r_def)
  end function damping_weighted
end module sweep_support_mod
"""


# The same module with the inner interface written over kinds PSyclone cannot
# reduce to a value: `r_solver` and `r_tran` are named by `constants_mod`,
# which these tests do not provide. Two names it cannot resolve may hold the
# same value or different ones, so both specifics match the actual weakly and
# the call is ambiguous rather than settled. It is the shape of a sibling
# that cannot be absorbed, reached through a generic rather than by name.
_UNSETTLED_GENERIC_MODULE = """
module sweep_support_mod
  use constants_mod, only : i_def, r_def, r_solver, r_tran
  implicit none
  private
  public :: sweep_column
  interface sweep_column
    module procedure sweep_column_damped, sweep_column_plain
  end interface sweep_column
  interface damping
    module procedure damping_solver, damping_tran
  end interface damping
contains
  subroutine sweep_column_plain(n, source, result)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    integer(kind=i_def) :: j
    result(n) = source(n) * damping(source(n))
    do j = n - 1, 1, -1
      result(j) = result(j + 1) - source(j) * damping(source(j))
    end do
  end subroutine sweep_column_plain
  subroutine sweep_column_damped(n, source, result, weight)
    integer(kind=i_def), intent(in) :: n
    real(kind=r_def), dimension(n), intent(in) :: source
    real(kind=r_def), dimension(n), intent(inout) :: result
    real(kind=r_def), intent(in) :: weight
    result(n) = source(n) * weight
  end subroutine sweep_column_damped
  function damping_solver(level) result(factor)
    real(kind=r_solver), intent(in) :: level
    real(kind=r_solver) :: factor
    factor = 1.0_r_solver / (level + 3.0_r_solver)
  end function damping_solver
  function damping_tran(level) result(factor)
    real(kind=r_tran), intent(in) :: level
    real(kind=r_tran) :: factor
    factor = 1.0_r_tran / (level + 3.0_r_tran)
  end function damping_tran
end module sweep_support_mod
"""


def _generic_invoke(tmp_path, module_source):
    """Build an invoke whose kernel calls a generic of ``module_source``.

    The kernel is the one Task E8 wrote for a callee that calls its own
    module: it ``use``s ``sweep_column`` and calls it with three arguments,
    which is what makes the outer call a generic to settle as well as the
    inner one.

    :param tmp_path: the directory the sources are written to.
    :type tmp_path: :py:class:`pathlib.Path`
    :param str module_source: the helper module the kernel ``use``s.

    :returns: as :py:func:`lfric_kokkos_sources._invoke` does.
    :rtype: Tuple[:py:class:`psyclone.psyGen.PSy`,
        :py:class:`psyclone.domain.lfric.LFRicLoop`,
        :py:class:`psyclone.domain.lfric.LFRicKern`]
    """
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _SIBLING_CALLEE_KERNEL,
                   extra={"sweep_support_mod": module_source})


def test_a_specific_settles_its_own_module_generic(
        tmp_path, clear_module_manager_instance):
    """A specific's call to a generic of its own module is settled.

    `pointwise_coordinate_jacobian`'s shape: the kernel calls a generic, and
    each specific behind it calls a second generic of the same module.
    ``KernelModuleInlineTrans`` validates every specific of the interface,
    not the one the arguments select, so every one of them has to be made
    self-contained before any of them can travel -- and each selects a
    different specific of the inner interface.

    The interfaces name the unselected specific first, so a body chosen by
    order rather than by argument would be the four-argument one, and its
    literal would show in the C++.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, loop, kernel = _generic_invoke(tmp_path, _GENERIC_SIBLING_MODULE)

    cpp = LFRicKokkosTrans().apply(loop)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    assert "damping" not in cpp
    # Twice, once per call site, and the weighted specific's literal nowhere.
    assert cpp.count("1.0 / ") == 2
    assert "9.0" not in cpp


def test_an_unsettled_inner_generic_leaves_the_call(
        tmp_path, clear_module_manager_instance):
    """A generic the arguments do not settle is left where the file put it.

    Both specifics of the inner interface are written over a kind PSyclone
    cannot reduce to a value, so neither is preferred and choosing between
    them would be guessing. The call is left, and the reader is told what
    that leaves them with: the callee names something declared beside it in
    its own module. The message is about the interface rather than about
    either specific, because the interface is what the body names.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, loop, _ = _generic_invoke(tmp_path, _UNSETTLED_GENERIC_MODULE)

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    assert ("contains accesses to 'damping' which is declared in the callee "
            "module scope") in message


def test_a_call_to_another_module_is_not_a_sibling(fortran_reader):
    """A call the callee makes to a routine of another module is left.

    Only a call whose target is written beside the callee is absorbed into
    it, because only such a call is invisible to the caller once the callee
    travels. A plain name imported from elsewhere travels with its import,
    so ``_sibling_called`` says there is no sibling to settle.
    """
    container = fortran_reader.psyir_from_source("""
    module sweep_support_mod
      use elsewhere_mod, only : tidy
      implicit none
    contains
      subroutine sweep_column_plain(n)
        integer, intent(in) :: n
        call tidy(n)
      end subroutine sweep_column_plain
    end module sweep_support_mod
    """).children[0]

    call = container.walk(Call)[0]

    assert LFRicKokkosTrans._sibling_called(container, call) is None
