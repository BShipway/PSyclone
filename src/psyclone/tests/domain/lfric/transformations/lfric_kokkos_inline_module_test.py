# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for a callee that reads the names of its own module."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LOCAL_ALGORITHM, _LOCAL_KERNEL, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.parse import ModuleManager
from psyclone.psyir.nodes import Call, IntrinsicCall, Routine
from psyclone.psyir.symbols import RoutineSymbol
from psyclone.psyir.transformations import TransformationError


# LFRic's `held_suarez_forcings_mod` and `sci_chi_transform_mod` in one
# module. `KF` and `KA` are private parameters, the second declared in terms
# of the first; `stretch` and `to_stretch` are the run-time state
# `init_chi_transforms` sets, as the `lfric_core` edit declares it: public
# and protected. `hidden` is that state as the file declared it before the
# edit, private, and nothing can carry it.
_DAMPING_MODULE = """
module damping_mod
  use constants_mod, only : i_def, r_def, l_def
  implicit none
  private
  real(kind=r_def), parameter :: KF = 1._r_def/86400._r_def
  real(kind=r_def), parameter :: KA = KF/40.0_r_def
  real(kind=r_def), public, parameter :: ks = 3.0_r_def
  real(kind=r_def), public, protected :: stretch = 1.0_r_def
  logical(kind=l_def), public, protected :: to_stretch
  real(kind=r_def) :: hidden
  public :: damp_column, scale_column, hidden_column
contains
  subroutine damp_column(levels, source, result)
    integer(kind=i_def), intent(in) :: levels
    real(kind=r_def), dimension(levels), intent(in) :: source
    real(kind=r_def), dimension(levels), intent(inout) :: result
    integer(kind=i_def) :: j
    do j = 1, levels
      result(j) = source(j) * KA + ks
    end do
  end subroutine damp_column
  subroutine scale_column(levels, source, result)
    integer(kind=i_def), intent(in) :: levels
    real(kind=r_def), dimension(levels), intent(in) :: source
    real(kind=r_def), dimension(levels), intent(inout) :: result
    integer(kind=i_def) :: j
    do j = 1, levels
      result(j) = source(j)
      if (to_stretch) result(j) = result(j) * stretch
    end do
  end subroutine scale_column
  subroutine hidden_column(levels, source, result)
    integer(kind=i_def), intent(in) :: levels
    real(kind=r_def), dimension(levels), intent(in) :: source
    real(kind=r_def), dimension(levels), intent(inout) :: result
    integer(kind=i_def) :: j
    do j = 1, levels
      result(j) = source(j) * hidden
    end do
  end subroutine hidden_column
end module damping_mod
"""


def _calling(callee):
    """Return the local kernel with its sweep replaced by a call.

    :param str callee: the routine of ``damping_mod`` the kernel calls.

    :returns: the kernel module's source.
    :rtype: str
    """
    return _LOCAL_KERNEL.replace(
        "  use kernel_mod, only : kernel_type",
        "  use kernel_mod, only : kernel_type\n"
        f"  use damping_mod, only : {callee}").replace(
        "    swept(nlayers) = partial(nlayers)\n"
        "    do k = nlayers - 1, 1, -1\n"
        "      swept(k) = swept(k + 1) - partial(k)\n"
        "    end do\n",
        f"    call {callee}(nlayers, partial, swept)\n")


def _damping_invoke(tmp_path, callee):
    """Build an invoke whose kernel calls a routine of ``damping_mod``.

    :param tmp_path: the directory the sources are written to.
    :type tmp_path: :py:class:`pathlib.Path`
    :param str callee: the routine of ``damping_mod`` the kernel calls.

    :returns: as :py:func:`lfric_kokkos_sources._invoke` does.
    :rtype: Tuple[:py:class:`psyclone.psyGen.PSy`,
        :py:class:`psyclone.domain.lfric.LFRicLoop`,
        :py:class:`psyclone.domain.lfric.LFRicKern`]
    """
    return _invoke(tmp_path, "column_solve", _LOCAL_ALGORITHM,
                   _calling(callee), extra={"damping_mod": _DAMPING_MODULE})


def _cached_names(callee):
    """Name what the cached module's routine declares, in table order.

    :param str callee: the routine of ``damping_mod`` to read.

    :returns: the names of the routine's own symbols.
    :rtype: List[str]
    """
    module = ModuleManager.get().get_module_info("damping_mod").get_psyir()
    routine = next(routine for routine in module.walk(Routine)
                   if routine.name == callee)
    return [symbol.name for symbol in routine.symbol_table.symbols]


def _no_call_left(kernel):
    """Say whether the kernel's schedule has had every call inlined.

    :param kernel: the kernel the transformation captured.
    :type kernel: :py:class:`psyclone.domain.lfric.LFRicKern`

    :returns: whether no call but an intrinsic is left in its body.
    :rtype: bool
    """
    schedule = LFRicKokkosTrans._schedule(kernel)
    return not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]


def test_a_private_parameter_travels_as_its_value(
        tmp_path, clear_module_manager_instance):
    """A callee's own module constants reach the region as values.

    `KA` is private to its module and declared in terms of `KF`, which is
    private too. The helper is given both as constants of its own, `KF`
    first, and the region carries the arithmetic rather than either name:
    the PSy layer imports neither, since the module does not let it. `ks`
    is public, so it is imported as any module constant is and passed by
    value.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    psy, loop, kernel = _damping_invoke(tmp_path, "damp_column")

    cpp = LFRicKokkosTrans().apply(loop)

    assert _no_call_left(kernel)
    assert "((1. / 86400.) / 40.0)" in cpp
    assert "const double ks" in cpp
    generated = str(psy.gen).lower()
    assert "use damping_mod, only : ks" in generated
    assert "kf" not in generated
    assert "ka" not in generated.replace("kokkos", "")


def test_a_public_protected_variable_travels_as_an_import(
        tmp_path, clear_module_manager_instance):
    """A callee's module state its module publishes is passed by value.

    This is LFRic's `chi2xyz` after the `lfric_core` edit: `to_stretch` and
    `stretch` are set once by an initialiser and only read afterwards, and
    `public, protected` lets any scope read them. The helper imports them
    from its own module, the PSy layer imports them too, and they cross the
    ABI as the by-value formals every module variable crosses it as.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    psy, loop, kernel = _damping_invoke(tmp_path, "scale_column")

    cpp = LFRicKokkosTrans().apply(loop)

    assert _no_call_left(kernel)
    assert "const double stretch" in cpp
    assert "const bool to_stretch" in cpp
    assert ("use damping_mod, only : stretch, to_stretch"
            in str(psy.gen).lower())


def test_a_private_variable_is_still_refused(
        tmp_path, clear_module_manager_instance):
    """A callee's private module state is refused in PSyclone's words.

    `hidden` is what `sci_chi_transform_mod`'s state was before the edit:
    a variable nothing outside its module may name, so there is no
    declaration to give the helper and nothing the PSy layer could pass.
    The refusal is the move's own, and names the variable.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, loop, _ = _damping_invoke(tmp_path, "hidden_column")

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'hidden_column'" in message
    assert ("routine 'hidden_column' contains accesses to 'hidden' which is "
            "declared in the callee module scope") in message


@pytest.mark.parametrize("callee", ["damp_column", "hidden_column"])
def test_the_cached_module_is_put_back(
        tmp_path, clear_module_manager_instance, callee):
    """The module the next kernel reads is the one its file declares.

    The declarations are added to the routine the `ModuleManager` holds,
    because that is the one the move reads, and taken out again when the
    move is over -- after a capture and after a refusal alike.
    """
    # The fixture is requested for its effect, not its value; the name is
    # too long to fit the disable-next its neighbours use on one line.
    # pylint: disable=unused-argument
    _, loop, _ = _damping_invoke(tmp_path, callee)

    try:
        LFRicKokkosTrans().validate(loop)
    except TransformationError:
        pass

    assert _cached_names(callee) == ["levels", "source", "result", "j"]


def test_a_callee_nothing_resolves_has_nothing_carried():
    """A call PSyclone cannot resolve is left for the caller to report.

    With no Container around it the call has no callee to read, and
    `get_callees` says so by raising. Nothing is carried and nothing is
    raised here: the refusal the caller then gets names the callee.
    """
    call = Call.create(RoutineSymbol("damp_column"))

    assert not LFRicKokkosTrans._carry_module_names(call)
