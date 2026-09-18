# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
# -----------------------------------------------------------------------------
"""Tests for LFRicKokkosInlineMixin: a PROTECTED actual argument."""

# pylint: disable=protected-access

import pytest

from lfric_kokkos_sources import _LOCAL_ALGORITHM, _LOCAL_KERNEL, _invoke

from psyclone.domain.lfric.transformations import LFRicKokkosTrans
from psyclone.psyir.nodes import Call, IntrinsicCall
from psyclone.psyir.transformations import TransformationError


# A configuration module of the shape LFRic generates one: variables a
# namelist reader sets and nothing else may assign to, which is what
# `protected` states. `coord_system` is the enumeration
# `sci_chi_transform_mod` hands to the native jacobian; the other two carry
# an attribute beside it, so that what is dropped can be told from what is
# not.
_COORD_CONFIG = """
module coord_config_mod
  use constants_mod, only : i_def, r_def
  implicit none
  integer(kind=i_def), public, protected :: coord_system = 1_i_def
  real(kind=r_def), public, protected, target :: scaled_radius = 1.0_r_def
  real(kind=r_def), public, pointer :: fitted_radius => null()
end module coord_config_mod
"""


def _calling_kernel(name, actual, declaration):
    """Return a kernel passing a module variable to a helper of its own.

    The helper is a module procedure of the kernel's own module, so nothing
    about reaching the callee is in question and the only thing that can
    refuse the call is the type of the actual argument.

    :param str name: the module variable the kernel imports.
    :param str actual: the expression the call passes for it.
    :param str declaration: the type the helper's formal is declared with.

    :returns: the kernel module source.
    :rtype: str

    """
    return _LOCAL_KERNEL.replace(
        "  use kernel_mod, only : kernel_type",
        "  use kernel_mod, only : kernel_type\n"
        f"  use coord_config_mod, only : {name}").replace(
        "    swept(nlayers) = partial(nlayers)\n"
        "    do k = nlayers - 1, 1, -1\n"
        "      swept(k) = swept(k + 1) - partial(k)\n"
        "    end do\n",
        f"    call sweep_column(nlayers, {actual}, partial, swept)\n").replace(
        "end module column_solve_kernel_mod",
        "  subroutine sweep_column(n, weight, source, result)\n"
        "    integer(kind=i_def), intent(in) :: n\n"
        f"    {declaration}, intent(in) :: weight\n"
        "    real(kind=r_def), dimension(n), intent(in) :: source\n"
        "    real(kind=r_def), dimension(n), intent(inout) :: result\n"
        "    integer(kind=i_def) :: j\n"
        "    result(n) = source(n)\n"
        "    do j = n - 1, 1, -1\n"
        "      result(j) = result(j + 1) - weight * source(j)\n"
        "    end do\n"
        "  end subroutine sweep_column\n"
        "end module column_solve_kernel_mod")


_PROTECTED_ACTUAL_KERNEL = _calling_kernel(
    "coord_system", "coord_system", "integer(kind=i_def)")


# The same call with a variable carrying `target` as well. `target` is an
# assertion about aliasing rather than about assignment, and it is not
# dropped here: a formal bound to it may be pointed at, which is a question
# about the callee's body and not about the call's arguments.
_PROTECTED_TARGET_ACTUAL_KERNEL = _calling_kernel(
    "scaled_radius", "scaled_radius", "real(kind=r_def)")


# The same call with a pointer, which carries no `protected` at all. What
# makes its declaration unmodelled is the pointer, so the refusal stands.
_POINTER_ACTUAL_KERNEL = _calling_kernel(
    "fitted_radius", "fitted_radius", "real(kind=r_def)")


@pytest.fixture(name="protected_actual_target")
# pylint: disable-next=unused-argument
def protected_actual_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke passing a PROTECTED module variable to a helper."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _PROTECTED_ACTUAL_KERNEL,
        extra={"coord_config_mod": _COORD_CONFIG})


@pytest.fixture(name="protected_target_actual_target")
# pylint: disable-next=unused-argument
def protected_target_actual_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke passing a PROTECTED, TARGET module variable."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM,
        _PROTECTED_TARGET_ACTUAL_KERNEL,
        extra={"coord_config_mod": _COORD_CONFIG})


@pytest.fixture(name="pointer_actual_target")
# pylint: disable-next=unused-argument
def pointer_actual_target_fixture(tmp_path, clear_module_manager_instance):
    """Create an invoke passing a POINTER module variable to a helper."""
    return _invoke(
        tmp_path, "column_solve", _LOCAL_ALGORITHM, _POINTER_ACTUAL_KERNEL,
        extra={"coord_config_mod": _COORD_CONFIG})


# ---------------------------------------------------------------------------
# Task G2: a PROTECTED actual argument is inlinable.
# ---------------------------------------------------------------------------


def test_protected_actual_is_inlined(protected_actual_target):
    """A helper called with a PROTECTED module variable is inlined.

    ``protected`` says that nothing outside the declaring module may assign
    to the variable. Passing it as an actual argument does not assign to it,
    and Fortran has already refused the call that would by requiring the
    formal to be ``intent(in)``, so the attribute has nothing to say about
    whether the call matches the routine. The PSyIR does not model it, so
    without the relaxation the argument arrives as an
    ``UnsupportedFortranType`` and the inliner reports no matching routine
    at all. This is the shape LFRic's configuration enumerations have, and
    it is what stood between the native jacobian and the card.
    """
    psy, loop, kernel = protected_actual_target

    cpp = LFRicKokkosTrans().apply(loop)
    fortran = str(psy.gen)

    schedule = LFRicKokkosTrans._schedule(kernel)
    assert not [call for call in schedule.walk(Call)
                if not isinstance(call, IntrinsicCall)]
    assert "sweep_column" not in cpp
    # The callee's statement, reading the caller's arrays and the imported
    # enumeration through the formals it was called with.
    assert ("swept((j - 1)) = (swept(((j + 1) - 1)) - "
            "(coord_system * partial((j - 1))));" in cpp)
    # It reaches the region as an argument, because only the PSy layer knows
    # what the namelist put in it.
    assert "const int coord_system" in cpp
    assert "use coord_config_mod, only : coord_system" in fortran


def test_protected_target_actual_is_still_refused(
        protected_target_actual_target):
    """A PROTECTED actual carrying TARGET as well is refused.

    Only ``protected`` is dropped, and only from a declaration that carries
    nothing else unmodelled. ``target`` is a statement about aliasing, which
    is a question about what the callee's body may do with the formal rather
    than about whether the argument matches it, so a declaration carrying
    both is left as the frontend produced it and refused as before.
    """
    _, loop, _ = protected_target_actual_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    assert "Argument type mismatch of call argument 'scaled_radius'" in message
    assert "PROTECTED, TARGET" in message


def test_pointer_actual_is_still_refused(pointer_actual_target):
    """An actual that is unmodelled for another reason is refused.

    A pointer is a name for storage that something else aims, which is not
    what makes a configuration variable's declaration unmodelled, and the
    relaxation passes over any declaration that does not say ``protected``.
    """
    _, loop, _ = pointer_actual_target

    with pytest.raises(TransformationError) as error:
        LFRicKokkosTrans().validate(loop)

    message = str(error.value)
    assert "cannot inline the call to 'sweep_column'" in message
    assert "Argument type mismatch of call argument 'fitted_radius'" in message
    assert "POINTER" in message
